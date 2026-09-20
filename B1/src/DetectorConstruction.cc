#include "DetectorConstruction.hh"
#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"

namespace B1
{
G4VPhysicalVolume* DetectorConstruction::Construct()
{
  auto* nist = G4NistManager::Instance();
  auto* air = nist->FindOrBuildMaterial("G4_AIR");
  auto* scint = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
  auto* aluminium = nist->FindOrBuildMaterial("G4_Al");
  const G4bool checkOverlaps = true;

  auto* worldSolid = new G4Box("World", 50*cm, 50*cm, 60*cm);
  auto* worldLV = new G4LogicalVolume(worldSolid, air, "World");
  auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0, checkOverlaps);

  auto makeDetector = [&](const G4String& name, G4double z, const G4Colour& colour) {
    auto* solid = new G4Box(name, 15*cm, 15*cm, 0.5*cm);
    auto* lv = new G4LogicalVolume(solid, scint, name);
    new G4PVPlacement(nullptr, {0, 0, z}, lv, name, worldLV, false, 0, checkOverlaps);
    lv->SetVisAttributes(new G4VisAttributes(colour));
  };
  makeDetector("UpstreamDetector1", -40*cm, G4Colour::Blue());
  makeDetector("UpstreamDetector2", -30*cm, G4Colour::Cyan());
  makeDetector("DownstreamDetector1", 30*cm, G4Colour::Green());
  makeDetector("DownstreamDetector2", 40*cm, G4Colour::Yellow());

  // U-shaped aluminium target. The opening faces +y so the shape is
  // immediately visible in the default oblique visualization view.
  // Overall footprint: 12 cm (x) x 12 cm (y) x 10 cm (z).
  const G4double outerHalfWidth = 6*cm;
  const G4double wallThickness = 3*cm;
  const G4double targetHalfDepth = 5*cm;
  const G4double sideHalfHeight = (12*cm - wallThickness)/2;
  const G4double sideCenterY = (-outerHalfWidth + wallThickness) + sideHalfHeight;

  auto* bottomSolid = new G4Box("UObjectBottom", outerHalfWidth,
                                wallThickness/2, targetHalfDepth);
  auto* sideSolid = new G4Box("UObjectSide", wallThickness/2,
                              sideHalfHeight, targetHalfDepth);
  auto* objectLV = new G4LogicalVolume(bottomSolid, aluminium, "UObjectBottom");
  auto* sideLV = new G4LogicalVolume(sideSolid, aluminium, "UObjectSide");

  new G4PVPlacement(nullptr, {0, -outerHalfWidth + wallThickness/2, 0},
                    objectLV, "UObjectBottom", worldLV, false, 0, checkOverlaps);
  new G4PVPlacement(nullptr, {-outerHalfWidth + wallThickness/2, sideCenterY, 0},
                    sideLV, "UObjectLeft", worldLV, false, 0, checkOverlaps);
  new G4PVPlacement(nullptr, {outerHalfWidth - wallThickness/2, sideCenterY, 0},
                    sideLV, "UObjectRight", worldLV, false, 1, checkOverlaps);

  objectLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));
  sideLV->SetVisAttributes(new G4VisAttributes(G4Colour(1., 0.15, 0.05, 0.75)));

  return worldPV;
}
}
