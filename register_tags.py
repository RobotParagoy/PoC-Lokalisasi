"""
register_tags.py
----------------
One-time script to register untracked AprilTags:
  1. Creates an Item in goods-service (port 3001)
  2. Creates an AprilTagMapping in blob-service (port 3002)

Edit UNTRACKED_TAGS below to match what you want to register, then run:
    python3 register_tags.py
"""

import jwt
import requests

GOODS_URL  = "http://localhost:3001"
BLOB_URL   = "http://localhost:3002"
JWT_SECRET = "change_me_in_production"

# ── Tags to register ──────────────────────────────────────────────────────────
# apriltag_id → { name, sku, category, description }
UNTRACKED_TAGS = {
    580: {"name": "Item 580", "sku": "TAG-580", "category": "tracked-item", "description": ""},
    581: {"name": "Item 581", "sku": "TAG-581", "category": "tracked-item", "description": ""},
    582: {"name": "Item 582", "sku": "TAG-582", "category": "tracked-item", "description": ""},
    583: {"name": "Item 583", "sku": "TAG-583", "category": "tracked-item", "description": ""},
    585: {"name": "Item 585", "sku": "TAG-585", "category": "tracked-item", "description": ""},
}


def make_token() -> str:
    return jwt.encode({"sub": "register_tags"}, JWT_SECRET, algorithm="HS256")


def create_item(token: str, apriltag_id: int, info: dict) -> int | None:
    """Create item in goods-service. Returns item_id on success."""
    resp = requests.post(
        f"{GOODS_URL}/items",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name":        info["name"],
            "sku":         info["sku"],
            "category":    info["category"],
            "description": info.get("description", ""),
            "quantity":    1,
        },
    )
    if resp.status_code == 409:
        print(f"  [SKIP] Item '{info['sku']}' already exists — fetching existing item_id")
        # try to look up by SKU if already created
        r2 = requests.get(
            f"{GOODS_URL}/items/sku/{info['sku']}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r2.ok:
            return r2.json()["data"]["item_id"]
        print(f"  [ERR] Could not fetch existing item for SKU {info['sku']}")
        return None
    if not resp.ok:
        print(f"  [ERR] Failed to create item for tag {apriltag_id}: {resp.text}")
        return None
    return resp.json()["data"]["item_id"]


def create_mapping(token: str, apriltag_id: int, item_id: int) -> bool:
    """Create apriltag mapping in blob-service."""
    resp = requests.post(
        f"{BLOB_URL}/apriltag-mappings",
        headers={"Authorization": f"Bearer {token}"},
        json={"apriltag_id": apriltag_id, "item_id": item_id},
    )
    if resp.status_code == 409:
        print(f"  [SKIP] Mapping for tag {apriltag_id} already exists")
        return True
    if not resp.ok:
        print(f"  [ERR] Failed to create mapping for tag {apriltag_id}: {resp.text}")
        return False
    return True


def main():
    token = make_token()
    print(f"Registering {len(UNTRACKED_TAGS)} tags...\n")

    for apriltag_id, info in UNTRACKED_TAGS.items():
        print(f"Tag {apriltag_id} — {info['name']}")

        item_id = create_item(token, apriltag_id, info)
        if item_id is None:
            continue
        print(f"  item_id = {item_id}")

        ok = create_mapping(token, apriltag_id, item_id)
        if ok:
            print(f"  mapping created  (apriltag {apriltag_id} → item {item_id})")

    print("\nDone. Restart main.py to pick up the new mappings.")


if __name__ == "__main__":
    main()
