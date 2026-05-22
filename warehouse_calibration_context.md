# Smart Warehouse Calibration and Vision Integration Context

This document provides a comprehensive technical guide for integrating 3D camera calibration and inverse ray projection into an existing real-time RTSP-based warehouse tracking pipeline. It covers the underlying geometry, step-by-step math, and structural Python code required to translate 2D image coordinates (bounding box centers) into precise, parallax-corrected 3D world coordinates on a warehouse grid system.

---

## 1. System Paradigm & Problem Statement

In a top-down warehouse tracking architecture, a central camera captures an overhead view of the entire operational floor. Tracking markers (such as QR codes or AprilTags) are attached to various components including Automated Guided Vehicles (AGVs), automated forklifts, static shelving units, and temporary floor pallets.

### The Parallax Challenge
Because a standard camera utilizes a perspective projection lens (modeled by pinhole camera equations), objects that are closer to the lens appear larger and are shifted outward radially toward the image edges relative to their true position on the floor plane. 

* **Flat Plane Constraints:** A standard 2D Homography matrix ($3 	imes 3$) assumes all entities reside on a single, fixed plane ($Z = 0$).
* **Multi-Height Interference:** Since an AGV's marker sits at height $h_{\text{robot}}$ and a shelf's marker sits at height $h_{\text{shelf}}$, passing raw pixel coordinates through a single floor-level homography introduces significant tracking offsets (parallax drift). 

To ensure exact localization, the system must decode the target's identity, look up its structural height parameter ($Z = h$), and intersect a mathematically unwarped 3D camera ray with that specific height plane.

---

## 2. Mathematical Foundation

The conversion relies on the inversion of the pinhole camera model, which establishes a formal link between a 3D world coordinate $\mathbf{X}_w = [X, Y, Z]^T$ and a 2D image pixel coordinate $\mathbf{x} = [u, v]^T$.

### Forward Camera Projection
$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \Big( R \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + t \Big)$$

Where:
* **$K$ (Intrinsic Matrix):** A $3 \times 3$ matrix encapsulating focal lengths ($f_x, f_y$) and the principal point ($c_x, c_y$).
* **$R$ (Rotation Matrix) & $t$ (Translation Vector):** Extrinsic parameters describing the camera's pose (orientation and offset) relative to the defined world origin ($X=0, Y=0, Z=0$).
* **$s$:** A scalar depth factor representing distance along the visual axis.

### Inverse Projection with Height Constraints
Every pixel coordinate defines an infinite linear ray in 3D space originating from the camera lens center $\mathbf{C}_w$. To extract the singular real-world point $(X, Y)$, we enforce a rigid boundary condition by explicitly providing the object's physical height: $Z = h$.

1. **Normalized Coordinates:** Extract lens distortion parameters via calibration, allowing transformation to a normalized image plane:
   $$\begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

2. **Ray Vector Derivation:** Rotate this vector from camera-local space into the absolute world frame using the transpose of the rotation matrix ($R^T$):
   $$\mathbf{v}_w = R^T \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix} = \begin{bmatrix} dx \\ dy \\ dz \end{bmatrix}$$

3. **Camera Center Extraction:** Calculate the physical position of the camera center in world coordinates:
   $$\mathbf{C}_w = -R^T t = \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

4. **Plane Intersection:** Solve for the exact scalar factor $s$ at which the ray passes through the targeted horizontal plane ($Z = h$):
   $$s = \frac{h - Z_c}{dz}$$

5. **Final Coordinate Realization:** Use $s$ to pin down the true $X$ and $Y$ offsets relative to the floor grid:
   $$X_{\text{world}} = X_c + s \cdot dx$$
   $$Y_{\text{world}} = Y_c + s \cdot dy$$

---

## 3. Implementation Workflow

Integrating this logic into a live application follows a two-tiered development sequence:

```
+--------------------------------------------------------------------------+
|                        OFFLINE CALIBRATION PHASE                         |
|                                                                          |
|  [Step A: Checkerboard Captures] ---> Compute Intrinsics (K) & DistCoeffs |
|  [Step B: Floor Reference Anchors] -> Run SolvePnP for Extrinsics (R, t) |
+--------------------------------------------------------------------------+
                                     |
                                     v Export Parameters to Config
+--------------------------------------------------------------------------+
|                        ONLINE TRACKING PIPELINE                          |
|                                                                          |
|  RTSP Stream --> Frame Capture --> Object Detection / Bounding Box       |
|                                                 |                        |
|   +---------------------------------------------+                        |
|   |                                                                      |
|   v (Identify Target Class / ID)                v (Calculate Center)     |
|  Lookup Known Height (h)                       Extract Pixel (u, v)      |
|   |                                             |                        |
|   +----------------------+----------------------+                        |
|                          |                                               |
|                          v                                               |
|           Apply Inverse Ray Projection Equation                          |
|                          |                                               |
|                          v                                               |
|        Map to Absolute Warehouse Grid Tile Location                      |
+--------------------------------------------------------------------------+
```

### Step 1: Intrinsic Calibration (Lens Flattening)
Capture 10–20 frames of a highly precise chessboard pattern placed at varying orientations throughout the field of view. Using `cv2.calibrateCamera()`, calculate the intrinsic parameters matrix $K$ and radial/tangential distortion parameters vector (`dist_coeffs`). This math step flattens optical warping (such as barrel or pincushion distortion).

### Step 2: Extrinsic Calibration (Spatial Alignment)
Map your coordinate spaces together. Mark out a series of fixed ground reference anchors on your warehouse floor layout (e.g., specific corners of tile intersections). Measure their true spatial positions in physical units (centimeters or meters) where $Z = 0$. Record their exact pixel centers $(u, v)$ from the camera feed. Pass these correspondences into `cv2.solvePnP()` to compute the exact 3D orientation vector (`rvec`) and translation vector (`tvec`) of the overhead lens relative to your grid origin.

### Step 3: Run-Time Tracking Correction
For every video frame ingested via the RTSP connection, compute the bounding box center of the item or robot. Use its identity or class tag to query its physical height layer, and project the 2D bounding box center out to its accurate position on the warehouse floor grid.

---

## 4. Production Python Implementation

The following production script demonstrates how to construct a robust `WarehouseCoordinateTransformer` class. This script handles the mathematical transformations, error checking, and object height registries, making it straightforward to drop directly into an active RTSP-based inference frame loop.

```python
import cv2
import numpy as np
import logging

# Configure tracking log output format
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
        
        # Structural asset directory mapping object types/IDs to physical height values (e.g., in cm)
        self._height_registry = {}
        
        logging.info("WarehouseCoordinateTransformer initialized successfully.")
        logging.info(f"Camera position resolved at World Coordinates: "
                     f"X={self.camera_world_pos[0][0]:.2f}, "
                     f"Y={self.camera_world_pos[1][0]:.2f}, "
                     f"Z={self.camera_world_pos[2][0]:.2f}")

    def register_object_height(self, object_class_or_id, height):
        """Registers a known physical height constraint for a specific class or ID."""
        self._height_registry[str(object_class_or_id)] = float(height)
        logging.info(f"Registered height constraint: '{object_class_or_id}' -> {height} cm")

    def get_height(self, object_class_or_id, default=0.0):
        """Retrieves the height constraint for an object from the directory."""
        return self._height_registry.get(str(object_class_or_id), default)

    def pixel_to_warehouse_world(self, u, v, target_height):
        """
        Projects a 2D bounding box center pixel coordinate back to its 3D position
        on the warehouse floor grid, accounting for its known height.
        
        Parameters:
        -----------
        u : float
            Horizontal pixel coordinate.
        v : float
            Vertical pixel coordinate.
        target_height : float
            The physical height (Z component) of the marker/bounding box.
            
        Returns:
        --------
        tuple (X, Y) or None
            Absolute coordinates relative to the warehouse grid origin.
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
            
            # Protect against parallel alignment or bad projections
            if abs(dz) < 1e-6:
                logging.warning(f"Ray for pixel ({u}, {v}) is parallel to the floor plane.")
                return None
                
            # Extract world positions of the camera center
            X_c, Y_c, Z_c = self.camera_world_pos[0][0], self.camera_world_pos[1][0], self.camera_world_pos[2][0]
            
            # 4. Compute tracking scaling value s based on intersection height plane (Z = target_height)
            s = (target_height - Z_c) / dz
            
            # 5. Derive the exact warehouse coordinate
            x_world = X_c + s * dx
            y_world = Y_c + s * dy
            
            return (x_world, y_world)
            
        except Exception as e:
            logging.error(f"Error calculating ray-plane intersection: {e}")
            return None

# --- PIPELINE INTEGRATION EXAMPLE ---
if __name__ == "__main__":
    print("--- Pipeline Configuration Mock Run ---")
    
    # 1. Provide calibration parameters (from your checkerboard and ground markers)
    # Mock parameters representing a camera mounted 2.5 meters above the ground looking downward
    mock_K = [[1000.0,    0.0,  640.0],
              [   0.0, 1000.0,  360.0],
              [   0.0,    0.0,    1.0]]
    mock_dist = [0.05, -0.1, 0.0, 0.0, 0.0]
    mock_rvec = [[0.05], [0.02], [0.0]]  # Slight lens tilt off axis
    mock_tvec = [[-10.0], [-20.0], [250.0]] # Ground offset relative to floor origin
    
    # 2. Instantiate core utility class
    transformer = WarehouseCoordinateTransformer(mock_K, mock_dist, mock_rvec, mock_tvec)
    
    # 3. Register your tracking heights based on inventory specs
    transformer.register_object_height("agv_heavy", height=18.5)   # Robot QR height
    transformer.register_object_height("tote_box", height=32.0)    # Box center height
    transformer.register_object_height("floor_tile", height=0.0)   # Flat ground context
    
    # 4. Simulation of RTSP tracking loop extraction
    mock_detections = [
        {"class_id": "agv_heavy", "bbox_center": (710.5, 420.0)},
        {"class_id": "tote_box",  "bbox_center": (850.0, 210.5)},
        {"class_id": "floor_tile", "bbox_center": (640.0, 360.0)}
    ]
    
    print("
--- Processing Detection Center Coordinates ---")
    for item in mock_detections:
        u, v = item["bbox_center"]
        cid = item["class_id"]
        
        # Retrieve registered target plane tracking level
        obj_height = transformer.get_height(cid)
        
        # Resolve true ground spatial position
        coords = transformer.pixel_to_warehouse_world(u, v, obj_height)
        
        if coords:
            print(f"Target Object Class: [{cid:10s}] | Center Pixel: ({u:5.1f}, {v:5.1f}) "
                  f"-> Absolute Warehouse Floor Position: X = {coords[0]:6.2f} cm, Y = {coords[1]:6.2f} cm")
