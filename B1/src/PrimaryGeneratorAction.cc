#include "PrimaryGeneratorAction.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4ios.hh"
#include "Randomize.hh"

#include <cmath>

namespace B1
{

namespace
{
// ============================================================================
// ===================== 你最常需要改的开关 =====================
// 在这里选择缪子入射模型。修改后需要重新编译，再运行 Geant4。
//
// Ideal（当前默认、论文基线）：
//   固定能量 4 GeV，全部沿 +z 方向平行入射。
//   优点是可重复、容易看清 U 型几何与 POCA 算法本身的效果。
//
// SemiRealistic（后续鲁棒性验证）：
//   随机入射位置、随机方位角、近似 cos²(theta) 的天顶角分布，
//   再使用 2--8 GeV 的简化能量范围。
//   它只是帮助理解“自然入射会怎样影响成像”的中间模型，不能称为
//   精确宇宙线缪子谱。正式贴近实验时再考虑 CRY 或 EcoMug。
//
// 只改下面这一行的 Ideal / SemiRealistic；不要同时改其他参数。
enum class MuonSourceMode
{
  Ideal,
  SemiRealistic
};

constexpr MuonSourceMode kMuonSourceMode = MuonSourceMode::Ideal;

constexpr G4double kSourceZ = -50. * cm;
constexpr G4double kSourceHalfWidth = 15. * cm;
constexpr G4double kSemiRealisticThetaMax = 45. * deg;
constexpr G4double kSemiRealisticMinEnergy = 2. * GeV;
constexpr G4double kSemiRealisticMaxEnergy = 8. * GeV;

// 对 p(theta) ∝ cos²(theta)·sin(theta) 进行反函数采样。
// 这里的 sin(theta) 来自球坐标的立体角元素，保证抽样分布正确。
G4double SampleCosineSquaredTheta()
{
  const auto cosThetaMax = std::cos(kSemiRealisticThetaMax);
  const auto cosThetaMaxCubed = cosThetaMax * cosThetaMax * cosThetaMax;
  const auto cosTheta = std::cbrt(
      cosThetaMaxCubed + G4UniformRand() * (1. - cosThetaMaxCubed));
  return std::acos(cosTheta);
}
}  // namespace

PrimaryGeneratorAction::PrimaryGeneratorAction()
{
  fParticleGun = new G4ParticleGun(1);
  auto* muon = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
  fParticleGun->SetParticleDefinition(muon);
  fParticleGun->SetParticleEnergy(4. * GeV);
  fParticleGun->SetParticleMomentumDirection({0., 0., 1.});

  if (kMuonSourceMode == MuonSourceMode::Ideal) {
    G4cout << "[Muon source] IDEAL mode: fixed 4 GeV, +z direction" << G4endl;
  }
  else {
    G4cout << "[Muon source] SEMI-REALISTIC mode: cos^2(theta), 2--8 GeV"
           << G4endl;
  }
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete fParticleGun; }

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  // 在上游源平面上均匀抽取入射位置。这样 x-y 平面不会预先偏向目标，
  // 最终的高散射区域才主要来自 U 型铝目标而不是源分布。
  const auto x = (G4UniformRand() * 2. - 1.) * kSourceHalfWidth;
  const auto y = (G4UniformRand() * 2. - 1.) * kSourceHalfWidth;

  if (kMuonSourceMode == MuonSourceMode::Ideal) {
    // 基线模式：每条缪子完全相同，只保留材料散射造成的轨迹变化。
    fParticleGun->SetParticleEnergy(4. * GeV);
    fParticleGun->SetParticleMomentumDirection({0., 0., 1.});
  }
  else {
    // 简化自然缪子模式：
    //   - 方位角 phi 在 0 到 2π 均匀；
    //   - 天顶角 theta 服从近似 cos²(theta) 分布；
    //   - 能量在 2--8 GeV 内均匀抽样。
    const auto theta = SampleCosineSquaredTheta();
    const auto phi = 2. * CLHEP::pi * G4UniformRand();
    const auto sinTheta = std::sin(theta);

    fParticleGun->SetParticleEnergy(
        kSemiRealisticMinEnergy
        + (kSemiRealisticMaxEnergy - kSemiRealisticMinEnergy) * G4UniformRand());
    fParticleGun->SetParticleMomentumDirection(
        G4ThreeVector(sinTheta * std::cos(phi),
                      sinTheta * std::sin(phi),
                      std::cos(theta)));
  }

  fParticleGun->SetParticlePosition({x, y, kSourceZ});
  fParticleGun->GeneratePrimaryVertex(event);
}
}
