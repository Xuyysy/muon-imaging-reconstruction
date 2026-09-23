#include "RunAction.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include <cstdlib>

namespace B1
{
void RunAction::BeginOfRunAction(const G4Run*)
{
  // Count only events that later satisfy the four-plane completeness check.
  fRecordedTracks = 0;
  // Random seeds are controlled externally for this introductory project;
  // disabling Geant4's automatic state files avoids unrelated output files.
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);
}
void RunAction::EndOfRunAction(const G4Run* run)
{
  // 不要把输出文件名写死：不同目标场景通过环境变量各自保存 CSV。
  const char* configuredOutput = std::getenv("MUON_TRACK_OUTPUT");
  const char* outputName = configuredOutput != nullptr ? configuredOutput
                                                    : "../../muon_analysis/muon_tracks.csv";
  G4cout << "\nMuon imaging run finished\n"
         << "  Generated events: " << run->GetNumberOfEvent() << "\n"
         << "  Complete tracks:  " << fRecordedTracks << "\n"
         << "  Output file:      " << outputName << "\n" << G4endl;
}
}
