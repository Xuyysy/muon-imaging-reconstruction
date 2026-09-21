#ifndef B1EventAction_h
#define B1EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

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
    void SetHit(const G4String& detector, G4double x, G4double y, G4double z);

  private:
    RunAction* fRunAction = nullptr;
    // Array order is fixed as: upstream 1/2, downstream 1/2. Keeping the
    // order explicit makes the CSV columns and PoCA reconstruction consistent.
    G4bool fHasHit[4] = {false, false, false, false};
    G4double fX[4] = {0., 0., 0., 0.};
    G4double fY[4] = {0., 0., 0., 0.};
    G4double fZ[4] = {0., 0., 0., 0.};
};
}

#endif
