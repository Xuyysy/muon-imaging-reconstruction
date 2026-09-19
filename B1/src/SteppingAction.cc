#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"

namespace B1
{
SteppingAction::SteppingAction(EventAction* eventAction) : fEventAction(eventAction) {}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  if (step->GetPreStepPoint()->GetStepStatus() != fGeomBoundary) return;
  auto* volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
  if (!volume) return;
  auto position = step->GetPreStepPoint()->GetPosition();
  auto name = volume->GetLogicalVolume()->GetName();

  if (name == "UpstreamDetector1" || name == "UpstreamDetector2" ||
      name == "DownstreamDetector1" || name == "DownstreamDetector2")
    fEventAction->SetHit(name, position.x(), position.y(), position.z());
}
}
