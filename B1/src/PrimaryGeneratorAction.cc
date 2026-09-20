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
// USER SWITCH: choose the incident-muon model here.
//
// Ideal:
//   Fixed 4 GeV muons, parallel to the +z direction.
//   This is the original baseline and is best for debugging the geometry.
//
// SemiRealistic:
//   Random impact position, random azimuth, an approximate cos^2(theta)
//   zenith-angle distribution, and a simple 2--8 GeV energy range.
//   This is an educational intermediate model, not a final cosmic-ray model.
//   For a publication-quality natural-muon spectrum, use CRY or EcoMug later.
//
// Change only the value below, then rebuild the project.
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

// Sample theta for p(theta) proportional to cos^2(theta) sin(theta).
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
  // Shoot muons uniformly across the upstream source plane.
  // The central object will therefore appear as a high-scattering region.
  const auto x = (G4UniformRand() * 2. - 1.) * kSourceHalfWidth;
  const auto y = (G4UniformRand() * 2. - 1.) * kSourceHalfWidth;

  if (kMuonSourceMode == MuonSourceMode::Ideal) {
    // Baseline mode: keep the beam parallel and mono-energetic.
    fParticleGun->SetParticleEnergy(4. * GeV);
    fParticleGun->SetParticleMomentumDirection({0., 0., 1.});
  }
  else {
    // Educational natural-muon approximation:
    //   - azimuth is uniform;
    //   - zenith angle follows an approximate cos^2(theta) law;
    //   - energy is uniform in a simple 2--8 GeV interval.
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
