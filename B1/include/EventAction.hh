#ifndef B1EventAction_h
#define B1EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

#include <fstream>
#include <string>
#include <vector>

namespace B1
{
class RunAction;

class EventAction : public G4UserEventAction
{
  public:
    explicit EventAction(RunAction* runAction);
    ~EventAction() override = default;
    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;
    void ObserveTrack(G4int pdgId, G4int trackId, G4int parentId,
                      G4double globalTime);
    void SetHit(const G4String& detector, G4double x, G4double y, G4double z,
                G4int pdgId, G4int trackId, G4int parentId,
                G4double globalTime);
    void RecordBoundaryEntry(const G4String& detector, G4double x, G4double y,
                             G4double z, G4int pdgId, G4int trackId,
                             G4int parentId, G4double globalTime);

  private:
    struct HitRecord
    {
      G4bool present = false;
      G4double x = 0.;
      G4double y = 0.;
      G4double z = 0.;
      G4int pdgId = 0;
      G4int trackId = -1;
      G4int parentId = -1;
      G4double globalTime = 0.;
    };

    struct BoundaryRecord
    {
      G4String detector;
      G4double x = 0.;
      G4double y = 0.;
      G4double z = 0.;
      G4int pdgId = 0;
      G4int trackId = -1;
      G4int parentId = -1;
      G4double globalTime = 0.;
      G4int processingOrder = 0;
    };

    RunAction* fRunAction = nullptr;
    // Array order is fixed as: upstream 1/2, downstream 1/2. Keeping the
    // order explicit makes the CSV columns and PoCA reconstruction consistent.
    G4bool fHasHit[4] = {false, false, false, false};
    G4double fX[4] = {0., 0., 0., 0.};
    G4double fY[4] = {0., 0., 0., 0.};
    G4double fZ[4] = {0., 0., 0., 0.};
    HitRecord fSelectedHit[4];
    HitRecord fPrimaryHit[4];
    std::vector<G4int> fPrimaryMuonTrackIds;
    G4double fPrimaryMuonGlobalTime = 0.;
    std::vector<BoundaryRecord> fBoundaryRecords;
    std::ofstream fDiagnosticEvents;
    std::ofstream fDiagnosticBoundaries;
    G4int fBoundaryProcessingOrder = 0;
};
}

#endif
