import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WarehouseCoordinateTransformer:
    def __init__(self, intrinsic_matrix, dist_coeffs, rvec, tvec):
        """
        Initializes the coordinate transformer with camera calibration parameters.
        
        Parameters:
        -----------
        intrinsic_matrix : np.ndarray
            3x3 Camera matrix K.
        dist_coeffs : np.ndarray
            Vector of lens distortion coefficients (k1, k2, p1, p2, k3).
        rvec : np.ndarray
            3x1 Rotation vector from extrinsic calibration.
        tvec : np.ndarray
            3x1 Translation vector from extrinsic calibration.
        """
        self.K = np.array(intrinsic_matrix, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        
        # Convert rotation vector to 3x3 rotation matrix
        R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
        self.R_T = R.T
        
        # Precompute the camera's structural position in absolute world coordinates
        self.tvec = np.array(tvec, dtype=np.float64)
        self.camera_world_pos = -self.R_T @ self.tvec
        
        # Structural asset directory mapping object types/IDs to physical height values
        self._height_registry = {}
        
        logging.info("WarehouseCoordinateTransformer initialized.")

    @classmethod
    def from_quad(cls, quad, K, grid_cols, grid_rows, tile_size=1.0):
        """
        Construct a 3D transformer dynamically from the current 2D field quad.
        Assuming the frame has already been undistorted, dist_coeffs = 0.
        
        Parameters:
        -----------
        quad : list
            The FIELD_QUAD pixel coordinates [TL, TR, BR, BL]
        K : np.ndarray
            The actual or estimated intrinsic matrix for the *undistorted* image.
        grid_cols, grid_rows : int
            The layout of the grid.
        tile_size : float
            Physical dimension (e.g., in cm or meters) of a single grid cell.
        """
        # Define the exact physical locations of the 4 grid corners on the floor (Z=0)
        object_points = np.array([
            [0, 0, 0],
            [grid_cols * tile_size, 0, 0],
            [grid_cols * tile_size, grid_rows * tile_size, 0],
            [0, grid_rows * tile_size, 0]
        ], dtype=np.float64)
        
        image_points = np.array(quad, dtype=np.float64)
        dist_coeffs = np.zeros(5, dtype=np.float64)
        
        # Use solvePnP to find the rotation/translation of the camera relative to this floor grid
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success:
            raise ValueError("solvePnP failed to find camera pose from quad!")
            
        return cls(K, dist_coeffs, rvec, tvec)

    def register_object_height(self, object_class_or_id, height):
        """Registers a known physical height constraint for a specific class or ID."""
        self._height_registry[str(object_class_or_id)] = float(height)

    def get_height(self, object_class_or_id, default=0.0):
        """Retrieves the height constraint for an object from the directory."""
        return self._height_registry.get(str(object_class_or_id), default)

    def pixel_to_warehouse_world(self, u, v, target_height):
        """
        Projects a 2D bounding box center pixel coordinate back to its 3D position
        on the warehouse floor grid, accounting for its known height.
        
        Returns:
        --------
        tuple (X, Y) Absolute coordinates relative to the warehouse grid origin.
        """
        try:
            # 1. Unwarp pixel point to remove lens curvature distortions
            pixel_point = np.array([[[u, v]]], dtype=np.float64)
            undistorted = cv2.undistortPoints(pixel_point, self.K, self.dist_coeffs)
            x_n, y_n = undistorted[0][0][0], undistorted[0][0][1]
            
            # 2. Build directional ray vector in camera coordinate frame
            ray_camera = np.array([[x_n], [y_n], [1.0]], dtype=np.float64)
            
            # 3. Rotate ray direction vector into the world grid space
            ray_world = self.R_T @ ray_camera
            dx, dy, dz = ray_world[0][0], ray_world[1][0], ray_world[2][0]
            
            if abs(dz) < 1e-6:
                return None
                
            # Extract world positions of the camera center
            X_c, Y_c, Z_c = self.camera_world_pos[0][0], self.camera_world_pos[1][0], self.camera_world_pos[2][0]
            
            # The world grid Z-axis direction depends on the order of corner points.
            # Usually, X=Right, Y=Down means Z points INTO the floor. Thus Z_c is negative.
            # We want the target plane to be shifted from the floor (0) TOWARDS the camera by target_height.
            z_dir = 1.0 if Z_c > 0 else -1.0
            world_z_plane = z_dir * abs(target_height)
            
            # 4. Compute tracking scaling value s based on intersection height plane
            s = (world_z_plane - Z_c) / dz
            
            # 5. Derive the exact warehouse coordinate
            x_world = X_c + s * dx
            y_world = Y_c + s * dy
            
            return (x_world, y_world)
            
        except Exception as e:
            logging.error(f"Error calculating ray-plane intersection: {e}")
            return None
