#include "PrimaryGeneratorAction.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

namespace B1
{
PrimaryGeneratorAction::PrimaryGeneratorAction()
{
  fParticleGun = new G4ParticleGun(1);
  auto* muon = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
  fParticleGun->SetParticleDefinition(muon);
  fParticleGun->SetParticleEnergy(4*GeV);
  fParticleGun->SetParticleMomentumDirection({0., 0., 1.});
  // Shoot muons uniformly across the upstream detector plane.
  // The central object will therefore appear as a high-scattering region.
}
PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete fParticleGun; }
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  const auto x = (G4UniformRand() - 0.5) * 30*cm;
  const auto y = (G4UniformRand() - 0.5) * 30*cm;
  fParticleGun->SetParticlePosition({x, y, -50*cm});
  fParticleGun->GeneratePrimaryVertex(event);
}
}
