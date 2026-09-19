#include "RunAction.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"

namespace B1
{
void RunAction::BeginOfRunAction(const G4Run*)
{
  fRecordedTracks = 0;
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);
}
void RunAction::EndOfRunAction(const G4Run* run)
{
  G4cout << "\nMuon imaging run finished\n"
         << "  Generated events: " << run->GetNumberOfEvent() << "\n"
         << "  Complete tracks:  " << fRecordedTracks << "\n"
         << "  Output file:      muon_tracks.csv\n" << G4endl;
}
}
