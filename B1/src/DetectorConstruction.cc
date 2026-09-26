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
//   l_shape：从未参与以上规则设计的 L 型铝目标，用于留出形状验证。
//   double_block：两块分离的铝目标；用于冻结方法后的新形状验证。
//   t_shape：预注册的新 T 型铝目标；只用于冻结一致性细化规则的检验。
//   frame：新的闭合方框铝目标；只用于冻结 H/A/P 深度门的独立验证。
//   plus_iron：铁制十字目标；只用于噪声底回退规则的未见验证。
//   plus_al：与 plus_iron 完全相同的十字几何，只将目标材料换成铝。
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
  auto* iron = nist->FindOrBuildMaterial("G4_Fe");
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

  // 留出验证场景：L 型目标与所有 U 型训练/探索场景不同。
  // 两个互不重叠的长方体组成外轮廓 120 mm × 120 mm 的 L：
  //   底边 x=[-60,60], y=[-60,-30] mm；
  //   左臂 x=[-60,-30], y=[-30,60] mm；z 均为 [-50,50] mm。
  // 仅改变目标形状，铝材料、厚度、探测器和源保持原设置。
  const char* scenario = std::getenv("MUON_TARGET_SCENARIO");
  if (scenario != nullptr && std::string(scenario) == "l_shape") {
    auto* bottom = new G4Box("LObjectBottom", 6*cm, 1.5*cm, 5*cm);
    auto* arm = new G4Box("LObjectArm", 1.5*cm, 4.5*cm, 5*cm);
    auto* bottomLV = new G4LogicalVolume(bottom, aluminium, "LObjectBottom");
    auto* armLV = new G4LogicalVolume(arm, aluminium, "LObjectArm");
    new G4PVPlacement(nullptr, {0, -4.5*cm, 0}, bottomLV,
                      "LObjectBottom", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {-4.5*cm, 1.5*cm, 0}, armLV,
                      "LObjectArm", worldLV, false, 0, checkOverlaps);
    bottomLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    armLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    G4cout << "[Target] held-out L-shaped aluminium target, 120 mm footprint"
           << G4endl;
    return worldPV;
  }

  // 独立验证场景：两块完全分离、相同材料和厚度的铝块。
  // 左块 x=[-65,-15] mm，右块 x=[15,65] mm；两块 y=[-30,30] mm，
  // z=[-50,50] mm。中间 30 mm 空隙是检验轮廓伪影的关键区域。
  // 仅改变目标几何；四层探测器、材料和入射源均保持原样。
  if (scenario != nullptr && std::string(scenario) == "double_block") {
    auto* block = new G4Box("DoubleBlockSolid", 2.5*cm, 3.*cm, 5.*cm);
    auto* blockLV = new G4LogicalVolume(block, aluminium, "DoubleBlockAluminium");
    new G4PVPlacement(nullptr, {-4.*cm, 0, 0}, blockLV,
                      "DoubleBlockLeft", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {4.*cm, 0, 0}, blockLV,
                      "DoubleBlockRight", worldLV, false, 1, checkOverlaps);
    blockLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    G4cout << "[Target] independent double-block aluminium scene, 30 mm gap"
           << G4endl;
    return worldPV;
  }

  // 新的 T 型独立场景，几何数值在生成数据前已记录于
  // muon_analysis/CONFIDENCE_REFINEMENT_PROTOCOL.md。
  // 顶横梁 x=[-60,60], y=[20,50] mm；中柱 x=[-15,15], y=[-50,20] mm。
  // 两部分仅在 y=20 mm 处接触而不重叠，z 均为 [-50,50] mm。
  if (scenario != nullptr && std::string(scenario) == "t_shape") {
    auto* bar = new G4Box("TObjectBar", 6.*cm, 1.5*cm, 5.*cm);
    auto* stem = new G4Box("TObjectStem", 1.5*cm, 3.5*cm, 5.*cm);
    auto* barLV = new G4LogicalVolume(bar, aluminium, "TObjectBar");
    auto* stemLV = new G4LogicalVolume(stem, aluminium, "TObjectStem");
    new G4PVPlacement(nullptr, {0, 3.5*cm, 0}, barLV,
                      "TObjectBar", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {0, -1.5*cm, 0}, stemLV,
                      "TObjectStem", worldLV, false, 0, checkOverlaps);
    barLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    stemLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    G4cout << "[Target] preregistered T-shaped aluminium target" << G4endl;
    return worldPV;
  }

  // 按 FRAME_Z_GATE_VALIDATION_PROTOCOL.md 预先冻结的闭合方框。
  // 上下横梁：x=[-60,60]、y=[-60,-40] / [40,60] mm；
  // 左右纵梁：x=[-60,-40] / [40,60]、y=[-40,40] mm。
  // 四块在边界处相接但体积不重叠；中心开孔为 80 mm × 80 mm。
  // z 厚度与旧形状相同，以隔离 XY 拓扑改变的影响。
  if (scenario != nullptr && std::string(scenario) == "frame") {
    auto* horizontal = new G4Box("FrameHorizontal", 6.*cm, 1.*cm, 5.*cm);
    auto* vertical = new G4Box("FrameVertical", 1.*cm, 4.*cm, 5.*cm);
    auto* horizontalLV = new G4LogicalVolume(horizontal, aluminium, "FrameHorizontal");
    auto* verticalLV = new G4LogicalVolume(vertical, aluminium, "FrameVertical");
    new G4PVPlacement(nullptr, {0, -5.*cm, 0}, horizontalLV,
                      "FrameBottom", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {0, 5.*cm, 0}, horizontalLV,
                      "FrameTop", worldLV, false, 1, checkOverlaps);
    new G4PVPlacement(nullptr, {-5.*cm, 0, 0}, verticalLV,
                      "FrameLeft", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {5.*cm, 0, 0}, verticalLV,
                      "FrameRight", worldLV, false, 1, checkOverlaps);
    horizontalLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    verticalLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    G4cout << "[Target] preregistered closed-frame aluminium target" << G4endl;
    return worldPV;
  }

  // 事先写定于 NOISE_FLOOR_FALLBACK_PROTOCOL.md 的新材料/新形状。
  // 中央竖梁 x=[-15,15], y=[-60,60] mm；左右横臂 x=[-60,-15]
  // 与 [15,60], y=[-15,15] mm。三部分只在面上接触，不重叠。
  if (scenario != nullptr && std::string(scenario) == "plus_iron") {
    auto* stem = new G4Box("PlusIronStem", 1.5*cm, 6.*cm, 5.*cm);
    auto* arm = new G4Box("PlusIronArm", 2.25*cm, 1.5*cm, 5.*cm);
    auto* stemLV = new G4LogicalVolume(stem, iron, "PlusIronStem");
    auto* armLV = new G4LogicalVolume(arm, iron, "PlusIronArm");
    new G4PVPlacement(nullptr, {0, 0, 0}, stemLV,
                      "PlusIronStem", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {-3.75*cm, 0, 0}, armLV,
                      "PlusIronLeft", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {3.75*cm, 0, 0}, armLV,
                      "PlusIronRight", worldLV, false, 1, checkOverlaps);
    stemLV->SetVisAttributes(new G4VisAttributes(G4Colour(0.75, 0.25, 0.12, 0.8)));
    armLV->SetVisAttributes(new G4VisAttributes(G4Colour(0.75, 0.25, 0.12, 0.8)));
    G4cout << "[Target] preregistered plus-shaped iron target" << G4endl;
    return worldPV;
  }

  // 材料控制：严格沿用上面的十字体积与位置，只把 G4_Fe 改为 G4_Al。
  // 保留原铁目标构造分支，避免修改已存铁数据所用的几何定义。
  if (scenario != nullptr && std::string(scenario) == "plus_al") {
    auto* stem = new G4Box("PlusAlStem", 1.5*cm, 6.*cm, 5.*cm);
    auto* arm = new G4Box("PlusAlArm", 2.25*cm, 1.5*cm, 5.*cm);
    auto* stemLV = new G4LogicalVolume(stem, aluminium, "PlusAlStem");
    auto* armLV = new G4LogicalVolume(arm, aluminium, "PlusAlArm");
    new G4PVPlacement(nullptr, {0, 0, 0}, stemLV,
                      "PlusAlStem", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {-3.75*cm, 0, 0}, armLV,
                      "PlusAlLeft", worldLV, false, 0, checkOverlaps);
    new G4PVPlacement(nullptr, {3.75*cm, 0, 0}, armLV,
                      "PlusAlRight", worldLV, false, 1, checkOverlaps);
    stemLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    armLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
    G4cout << "[Target] plus-shaped aluminium material control" << G4endl;
    return worldPV;
  }

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
