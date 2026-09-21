#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"

namespace B1
{
SteppingAction::SteppingAction(EventAction* eventAction) : fEventAction(eventAction) {}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // Record only a volume-entry step. Recording every step would create many
  // hits in one plane and would no longer correspond to one detector reading.
  if (step->GetPreStepPoint()->GetStepStatus() != fGeomBoundary) return;
  auto* volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
  if (!volume) return;
  auto position = step->GetPreStepPoint()->GetPosition();
  auto name = volume->GetLogicalVolume()->GetName();

  // EventAction turns this ideal boundary position into one of the four
  // coordinates required for incoming/outgoing track reconstruction.
  if (name == "UpstreamDetector1" || name == "UpstreamDetector2" ||
      name == "DownstreamDetector1" || name == "DownstreamDetector2")
    fEventAction->SetHit(name, position.x(), position.y(), position.z());
}
}
