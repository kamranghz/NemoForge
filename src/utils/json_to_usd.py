"""
json_to_usd.py
Converts a SAGE-10K layout JSON to a physics-ready USD file.
No Isaac Sim required — uses pxr (usd-core) only.
"""
import json
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf, UsdShade

def convert_layout_to_usd(layout_json_path: str, output_usd_path: str) -> str:
    with open(layout_json_path) as f:
        layout = json.load(f)

    stage = Usd.Stage.CreateNew(output_usd_path)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Y")

    # Physics scene
    scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
    scene.CreateGravityMagnitudeAttr(9.81)

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr([
        Gf.Vec3f(-50,0,-50), Gf.Vec3f(50,0,-50),
        Gf.Vec3f(50,0,50),   Gf.Vec3f(-50,0,50)
    ])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0,1,2,3])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Objects from all rooms
    rooms = layout.get("rooms", [])
    obj_count = 0
    for room in rooms:
        for obj in room.get("objects", []):
            obj_id  = obj.get("id", f"obj_{obj_count}")
            safe_id = obj_id.replace("-", "_")
            pos     = obj.get("position", {"x":0,"y":0,"z":0})
            dims    = obj.get("dimensions",
                              {"width":1,"height":1,"length":1})
            mass    = obj.get("mass", 1.0)

            prim_path = f"/World/{safe_id}"
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)

            # Scale to object dimensions
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(
                pos["x"],
                pos.get("z", 0) + dims.get("height", 1) / 2,
                pos["y"]
            ))
            xform.AddScaleOp().Set(Gf.Vec3d(
                dims.get("width",  1),
                dims.get("height", 1),
                dims.get("length", 1)
            ))

            # Physics
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            rigid = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
            mass_api.CreateMassAttr(float(mass))

            obj_count += 1

    stage.GetRootLayer().Save()
    print(f"Saved {obj_count} objects to {output_usd_path}")
    return output_usd_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python json_to_usd.py <layout.json> <output.usd>")
        sys.exit(1)
    convert_layout_to_usd(sys.argv[1], sys.argv[2])
