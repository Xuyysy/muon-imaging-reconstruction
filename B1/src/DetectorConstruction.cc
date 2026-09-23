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
//   small：80 mm 外轮廓的中心 U 型目标；
//   medium_offset：预留给多尺度定位的独立验证场景（100 mm，偏心）。
//   small_offset：下一轮抗热点规则的独立验证（80 mm，双轴偏心）。
// 这样不需要反复改源代码、重新保存两套几何文件。
struct TargetConfiguration
{
  G4ThreeVector center;
  G4double outerHalfWidth;
  G4double wallThickness;
};

TargetConfiguration TargetConfigurationFromEnvironment()
{
  const char* scenario = std::getenv("MUON_TARGET_SCENARIO");
  if (scenario != nullptr && std::string(scenario) == "offcenter") {
    // 右侧壁最大 x 为 85 mm、底边最小 y 为 -75 mm，完整落在当前 ±90 mm 成像区内。
    return {{25.*mm, -15.*mm, 0.}, 6.*cm, 3.*cm};
  }
  if (scenario != nullptr && std::string(scenario) == "small") {
    // 尺寸泛化验证：80 mm x 80 mm 外轮廓、20 mm 壁厚，中心保持在原点。
    return {{0., 0., 0.}, 4.*cm, 2.*cm};
  }
  if (scenario != nullptr && std::string(scenario) == "medium_offset") {
    // 新场景不参与定位规则设计；偏移方向与旧 offcenter 相反。
    // 外轮廓 x=[-70,30]、y=[-35,65] mm，位于 ±90 mm 成像区域内。
    return {{-20.*mm, 15.*mm, 0.}, 5.*cm, 2.5*cm};
  }
  if (scenario != nullptr && std::string(scenario) == "small_offset") {
    // 小目标既不在几何中心，也不在上一轮的偏移方向。
    // 外轮廓 x=[-10,70]、y=[-20,60] mm，仍完整位于成像区域内。
    return {{30.*mm, 20.*mm, 0.}, 4.*cm, 2.*cm};
  }
  return {{0., 0., 0.}, 6.*cm, 3.*cm};
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
  const auto targetConfig = TargetConfigurationFromEnvironment();
  const G4double outerHalfWidth = targetConfig.outerHalfWidth;
  const G4double wallThickness = targetConfig.wallThickness;
  const G4double targetHalfDepth = 5*cm;
  // 侧壁的全高度应随外轮廓尺寸变化：outer size - bottom-wall thickness。
  // 原中心场景中 2*outerHalfWidth 恰好是 12 cm；写成通用表达式后，
  // small 场景也会得到正确的 80 mm U 型外轮廓。
  const G4double sideHalfHeight = (2.*outerHalfWidth - wallThickness)/2;
  const G4double sideCenterY = (-outerHalfWidth + wallThickness) + sideHalfHeight;
  const auto targetCenter = targetConfig.center;

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
         << targetCenter.z()/mm << ") mm, outer width = "
         << 2.*outerHalfWidth/mm << " mm, wall thickness = "
         << wallThickness/mm << " mm" << G4endl;

  objectLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
  sideLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));

  return worldPV;
}
}
