#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4TouchableHandle.hh"
#include "G4VPhysicalVolume.hh"

namespace B1
{
SteppingAction::SteppingAction(EventAction* eventAction) : fEventAction(eventAction) {}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  const auto* track = step->GetTrack();
  const auto pdgId = track->GetDefinition()->GetPDGEncoding();
  const auto trackId = track->GetTrackID();
  const auto parentId = track->GetParentID();
  fEventAction->ObserveTrack(pdgId, trackId, parentId,
                             step->GetPreStepPoint()->GetGlobalTime());

  // Independently record every actual entry into any of the four detector
  // volumes. The post-step point is the entered volume; Global Time lets the
  // audit compare physical chronology with Geant4 processing order.
  const auto* postPoint = step->GetPostStepPoint();
  if (postPoint->GetStepStatus() == fGeomBoundary) {
    auto* enteredVolume = postPoint->GetTouchableHandle()->GetVolume();
    if (enteredVolume) {
      const auto enteredName = enteredVolume->GetLogicalVolume()->GetName();
      if (enteredName == "UpstreamDetector1" || enteredName == "UpstreamDetector2" ||
          enteredName == "DownstreamDetector1" || enteredName == "DownstreamDetector2") {
        const auto position = postPoint->GetPosition();
        fEventAction->RecordBoundaryEntry(
            enteredName, position.x(), position.y(), position.z(), pdgId,
            trackId, parentId, postPoint->GetGlobalTime());
      }
    }
  }

  // Record only a volume-entry step. Recording every step would create many
  // hits in one plane and would no longer correspond to one detector reading.
  if (step->GetPreStepPoint()->GetStepStatus() != fGeomBoundary) return;
  auto* volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
  if (!volume) return;
  auto position = step->GetPreStepPoint()->GetPosition();
  auto name = volume->GetLogicalVolume()->GetName();

  // EventAction turns this ideal boundary position into one of the four
  // coordinates required for incoming/outgoing track reconstruction. Keep
  // this exact pre-step selection rule for the baseline; identity is logged
  // alongside it but never changes which hit wins.
  if (name == "UpstreamDetector1" || name == "UpstreamDetector2" ||
      name == "DownstreamDetector1" || name == "DownstreamDetector2")
    fEventAction->SetHit(name, position.x(), position.y(), position.z(),
                         pdgId, trackId, parentId,
                         step->GetPreStepPoint()->GetGlobalTime());
}
}
