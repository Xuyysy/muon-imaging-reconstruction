#include "DetectorConstruction.hh"
#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"

#include <cstdlib>
#include <string>

namespace B1
{
namespace
{
// 目标场景由环境变量 MUON_TARGET_SCENARIO 控制：
//   默认或 centered：中心 U 型目标，保持现有 PoCA 基线不变；
//   offcenter：向 +x、-y 偏移，用于验证算法是否只适用于中心目标。
// 这样不需要反复改源代码、重新保存两套几何文件。
G4ThreeVector TargetCenterFromEnvironment()
{
  const char* scenario = std::getenv("MUON_TARGET_SCENARIO");
  if (scenario != nullptr && std::string(scenario) == "offcenter") {
    // 右侧壁最大 x 为 85 mm、底边最小 y 为 -75 mm，完整落在当前 ±90 mm 成像区内。
    return {25.*mm, -15.*mm, 0.};
  }
  return {0., 0., 0.};
}
}  // namespace

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // This function defines every physical object in the Geant4 world.
  // Coordinate convention used throughout this project:
  //   origin = centre of the imaging region;
  //   +z     = nominal muon flight direction.
  auto* nist = G4NistManager::Instance();
  auto* air = nist->FindOrBuildMaterial("G4_AIR");
  auto* scint = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
  auto* aluminium = nist->FindOrBuildMaterial("G4_Al");
  const G4bool checkOverlaps = true;

  // The world only provides an air environment. Its half-lengths must be
  // larger than the source, all four detector planes, and the U target.
  auto* worldSolid = new G4Box("World", 50*cm, 50*cm, 60*cm);
  auto* worldLV = new G4LogicalVolume(worldSolid, air, "World");
  auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0, checkOverlaps);

  // A detector plane is represented as a thin plastic-scintillator slab.
  // In this introductory model we record the first boundary crossing as an
  // ideal hit; detector efficiency and spatial resolution can be introduced
  // later without changing the reconstruction workflow.
  auto makeDetector = [&](const G4String& name, G4double z, const G4Colour& colour) {
    auto* solid = new G4Box(name, 15*cm, 15*cm, 0.5*cm);
    auto* lv = new G4LogicalVolume(solid, scint, name);
    new G4PVPlacement(nullptr, {0, 0, z}, lv, name, worldLV, false, 0, checkOverlaps);
    lv->SetVisAttributes(new G4VisAttributes(colour));
  };
  // Two upstream planes fit the incoming track; two downstream planes fit
  // the outgoing track after the muon has crossed the target.
  makeDetector("UpstreamDetector1", -40*cm, G4Colour::Blue());
  makeDetector("UpstreamDetector2", -30*cm, G4Colour::Cyan());
  makeDetector("DownstreamDetector1", 30*cm, G4Colour::Green());
  makeDetector("DownstreamDetector2", 40*cm, G4Colour::Yellow());

  // U-shaped aluminium target. The opening faces +y so the shape is
  // immediately visible in the default oblique visualization view.
  // Overall footprint: 12 cm (x) x 12 cm (y) x 10 cm (z).
  // G4Box takes HALF lengths. Therefore:
  //   outerHalfWidth = 6 cm  -> 12 cm full x/y footprint;
  //   targetHalfDepth = 5 cm -> 10 cm full thickness along z.
  const G4double outerHalfWidth = 6*cm;
  const G4double wallThickness = 3*cm;
  const G4double targetHalfDepth = 5*cm;
  const G4double sideHalfHeight = (12*cm - wallThickness)/2;
  const G4double sideCenterY = (-outerHalfWidth + wallThickness) + sideHalfHeight;
  const auto targetCenter = TargetCenterFromEnvironment();

  auto* bottomSolid = new G4Box("UObjectBottom", outerHalfWidth,
                                wallThickness/2, targetHalfDepth);
  auto* sideSolid = new G4Box("UObjectSide", wallThickness/2,
                              sideHalfHeight, targetHalfDepth);
  auto* objectLV = new G4LogicalVolume(bottomSolid, aluminium, "UObjectBottom");
  auto* sideLV = new G4LogicalVolume(sideSolid, aluminium, "UObjectSide");

  // Build the U from three non-overlapping boxes: one bottom wall and two
  // side walls. The missing top wall makes the opening face +y.
  new G4PVPlacement(nullptr, targetCenter + G4ThreeVector(0, -outerHalfWidth + wallThickness/2, 0),
                    objectLV, "UObjectBottom", worldLV, false, 0, checkOverlaps);
  new G4PVPlacement(nullptr, targetCenter + G4ThreeVector(-outerHalfWidth + wallThickness/2, sideCenterY, 0),
                    sideLV, "UObjectLeft", worldLV, false, 0, checkOverlaps);
  new G4PVPlacement(nullptr, targetCenter + G4ThreeVector(outerHalfWidth - wallThickness/2, sideCenterY, 0),
                    sideLV, "UObjectRight", worldLV, false, 1, checkOverlaps);

  G4cout << "[Target] U-shaped aluminium centre: ("
         << targetCenter.x()/mm << ", " << targetCenter.y()/mm << ", "
         << targetCenter.z()/mm << ") mm" << G4endl;

  objectLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
  sideLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));

  return worldPV;
}
}
