#include "EventAction.hh"
#include "RunAction.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace B1
{

namespace
{
// The two incoming and two outgoing detector planes.
constexpr int kUpstream1 = 0;
constexpr int kUpstream2 = 1;
constexpr int kDownstream1 = 2;
constexpr int kDownstream2 = 3;

// 默认仍输出基线 CSV。设置 MUON_TRACK_OUTPUT 后可为不同验证场景
// 写入独立文件，从而绝不覆盖已有的 muon_tracks.csv。
std::string OutputPathFromEnvironment()
{
  const char* configured = std::getenv("MUON_TRACK_OUTPUT");
  return configured != nullptr ? configured : "../../muon_analysis/muon_tracks.csv";
}

}

EventAction::EventAction(RunAction* runAction) : fRunAction(runAction)
{
  const char* eventsPath = std::getenv("MUON_TRACK_DIAGNOSTICS_OUTPUT");
  const char* boundariesPath = std::getenv("MUON_TRACK_BOUNDARY_OUTPUT");
  if ((eventsPath == nullptr) != (boundariesPath == nullptr)) {
    throw std::runtime_error(
        "Set both MUON_TRACK_DIAGNOSTICS_OUTPUT and MUON_TRACK_BOUNDARY_OUTPUT, or neither.");
  }
  if (eventsPath != nullptr) {
    fDiagnosticEvents.open(eventsPath);
    fDiagnosticBoundaries.open(boundariesPath);
    if (!fDiagnosticEvents || !fDiagnosticBoundaries) {
      throw std::runtime_error("Could not open one or more track-identity diagnostic outputs.");
    }

    fDiagnosticEvents
        << "event_id,primary_candidate_count,primary_track_ids,primary_track_id,"
           "primary_pdg_id,primary_parent_id,primary_global_time_ns,"
           "current_complete,primary_complete";
    fDiagnosticBoundaries
        << "event_id,detector,processing_order,global_time_ns,pdg_id,track_id,"
           "parent_id,x_mm,y_mm,z_mm\n";
    for (int i = 0; i < 4; ++i) {
      const char* layer = i == 0 ? "up1" : i == 1 ? "up2" :
                          i == 2 ? "down1" : "down2";
      fDiagnosticEvents << ",sel_" << layer << "_track_id,sel_" << layer
                        << "_pdg_id,sel_" << layer << "_parent_id,sel_" << layer
                        << "_global_time_ns,sel_" << layer << "_x_mm,sel_" << layer
                        << "_y_mm,sel_" << layer << "_z_mm,primary_" << layer
                        << "_track_id,primary_" << layer << "_pdg_id,primary_"
                        << layer << "_parent_id,primary_" << layer
                        << "_global_time_ns,primary_" << layer << "_x_mm,primary_"
                        << layer << "_y_mm,primary_" << layer << "_z_mm";
    }
    fDiagnosticEvents << '\n';
  }
}

void EventAction::BeginOfEventAction(const G4Event*)
{
  // EventAction objects are reused between events. Clear the hit flags here
  // so a hit from the previous muon can never be written into this event.
  for (int i = 0; i < 4; ++i) fHasHit[i] = false;
  for (int i = 0; i < 4; ++i) {
    fSelectedHit[i] = HitRecord{};
    fPrimaryHit[i] = HitRecord{};
  }
  fPrimaryMuonTrackIds.clear();
  fPrimaryMuonGlobalTime = 0.;
  fBoundaryRecords.clear();
  fBoundaryProcessingOrder = 0;
}

void EventAction::ObserveTrack(G4int pdgId, G4int trackId, G4int parentId,
                               G4double globalTime)
{
  // PDG encoding 13 is the muon-minus. ParentID 0 identifies a primary
  // Geant4 track; its Track ID is discovered per event, never assumed.
  if (pdgId != 13 || parentId != 0) return;
  for (const auto candidateId : fPrimaryMuonTrackIds) {
    if (candidateId == trackId) return;
  }
  fPrimaryMuonTrackIds.push_back(trackId);
  if (fPrimaryMuonTrackIds.size() == 1) fPrimaryMuonGlobalTime = globalTime;
  if (fPrimaryMuonTrackIds.size() > 1) {
    // A later-discovered second primary μ− makes event-level identity
    // ambiguous. Invalidate any first-candidate boundary hits already stored.
    for (int i = 0; i < 4; ++i) fPrimaryHit[i] = HitRecord{};
  }
}

void EventAction::SetHit(const G4String& detector, G4double x, G4double y,
                         G4double z, G4int pdgId, G4int trackId,
                         G4int parentId, G4double globalTime)
{
  int index = -1;
  if (detector == "UpstreamDetector1") index = kUpstream1;
  if (detector == "UpstreamDetector2") index = kUpstream2;
  if (detector == "DownstreamDetector1") index = kDownstream1;
  if (detector == "DownstreamDetector2") index = kDownstream2;
  // Only the first crossing of each detector is used. This mimics one ideal
  // position measurement per plane and avoids duplicate entries from steps
  // inside the scintillator volume.
  if (index < 0 || fHasHit[index]) return;

  fHasHit[index] = true;
  fX[index] = x;
  fY[index] = y;
  fZ[index] = z;
  fSelectedHit[index] = {true, x, y, z, pdgId, trackId, parentId, globalTime};
}

void EventAction::RecordBoundaryEntry(const G4String& detector, G4double x,
                                      G4double y, G4double z, G4int pdgId,
                                      G4int trackId, G4int parentId,
                                      G4double globalTime)
{
  ++fBoundaryProcessingOrder;
  fBoundaryRecords.push_back(
      {detector, x, y, z, pdgId, trackId, parentId, globalTime,
       fBoundaryProcessingOrder});

  int index = -1;
  if (detector == "UpstreamDetector1") index = kUpstream1;
  if (detector == "UpstreamDetector2") index = kUpstream2;
  if (detector == "DownstreamDetector1") index = kDownstream1;
  if (detector == "DownstreamDetector2") index = kDownstream2;
  if (index < 0 || pdgId != 13 || parentId != 0 || fPrimaryMuonTrackIds.size() != 1 ||
      fPrimaryMuonTrackIds.front() != trackId || fPrimaryHit[index].present) {
    return;
  }
  fPrimaryHit[index] = {true, x, y, z, pdgId, trackId, parentId, globalTime};
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  if (fDiagnosticEvents.is_open()) {
    const auto eventId = event->GetEventID();
    const auto primaryComplete = fPrimaryMuonTrackIds.size() == 1 &&
                                 fPrimaryHit[kUpstream1].present &&
                                 fPrimaryHit[kUpstream2].present &&
                                 fPrimaryHit[kDownstream1].present &&
                                 fPrimaryHit[kDownstream2].present;
    const auto currentComplete = fHasHit[kUpstream1] && fHasHit[kUpstream2] &&
                                 fHasHit[kDownstream1] && fHasHit[kDownstream2];
    fDiagnosticEvents << eventId << ',' << fPrimaryMuonTrackIds.size() << ',';
    for (std::size_t i = 0; i < fPrimaryMuonTrackIds.size(); ++i) {
      if (i != 0) fDiagnosticEvents << ';';
      fDiagnosticEvents << fPrimaryMuonTrackIds[i];
    }
    fDiagnosticEvents << ',';
    if (fPrimaryMuonTrackIds.size() == 1) fDiagnosticEvents << fPrimaryMuonTrackIds.front();
    fDiagnosticEvents << ',';
    if (fPrimaryMuonTrackIds.size() == 1) fDiagnosticEvents << 13;
    fDiagnosticEvents << ',';
    if (fPrimaryMuonTrackIds.size() == 1) fDiagnosticEvents << 0;
    fDiagnosticEvents << ',';
    if (fPrimaryMuonTrackIds.size() == 1) fDiagnosticEvents << fPrimaryMuonGlobalTime / ns;
    fDiagnosticEvents << ',' << (currentComplete ? 1 : 0) << ','
                      << (primaryComplete ? 1 : 0);

    for (int i = 0; i < 4; ++i) {
      const auto writeHit = [this](const HitRecord& hit) {
        fDiagnosticEvents << ',';
        if (hit.present) fDiagnosticEvents << hit.trackId;
        fDiagnosticEvents << ',';
        if (hit.present) fDiagnosticEvents << hit.pdgId;
        fDiagnosticEvents << ',';
        if (hit.present) fDiagnosticEvents << hit.parentId;
        fDiagnosticEvents << ',';
        if (hit.present) fDiagnosticEvents << hit.globalTime / ns;
        fDiagnosticEvents << ',';
        if (hit.present) fDiagnosticEvents << hit.x / mm;
        fDiagnosticEvents << ',';
        if (hit.present) fDiagnosticEvents << hit.y / mm;
        fDiagnosticEvents << ',';
        if (hit.present) fDiagnosticEvents << hit.z / mm;
      };
      writeHit(fSelectedHit[i]);
      writeHit(fPrimaryHit[i]);
    }
    fDiagnosticEvents << '\n';

    for (const auto& entry : fBoundaryRecords) {
      fDiagnosticBoundaries << eventId << ',' << entry.detector << ','
                            << entry.processingOrder << ','
                            << entry.globalTime / ns << ',' << entry.pdgId << ','
                            << entry.trackId << ',' << entry.parentId << ','
                            << entry.x / mm << ',' << entry.y / mm << ','
                            << entry.z / mm << '\n';
    }
  }

  // The simulation is run from B1/build, so ../../ points to the project root
  // and keeps raw Geant4 output separate from the C++ source tree.
  static std::ofstream output(OutputPathFromEnvironment());
  static G4bool headerWritten = false;

  if (!headerWritten) {
    output << "event,up1_x_mm,up1_y_mm,up1_z_mm,"
              "up2_x_mm,up2_y_mm,up2_z_mm,"
              "down1_x_mm,down1_y_mm,down1_z_mm,"
              "down2_x_mm,down2_y_mm,down2_z_mm,"
              "poca_x_mm,poca_y_mm,poca_z_mm,"
              "theta_x_mrad,theta_y_mrad,theta_mrad\n";
    headerWritten = true;
  }

  // A PoCA event needs exactly two upstream and two downstream positions.
  // Incomplete tracks are deliberately ignored instead of being imputed.
  for (int i = 0; i < 4; ++i) {
    if (!fHasHit[i]) return;
  }

  const G4ThreeVector up1(fX[kUpstream1], fY[kUpstream1], fZ[kUpstream1]);
  const G4ThreeVector up2(fX[kUpstream2], fY[kUpstream2], fZ[kUpstream2]);
  const G4ThreeVector down1(fX[kDownstream1], fY[kDownstream1], fZ[kDownstream1]);
  const G4ThreeVector down2(fX[kDownstream2], fY[kDownstream2], fZ[kDownstream2]);

  // With two ideal hit positions on each side, the track fit is simply the
  // normalized vector from the first plane to the second plane.
  const auto incoming = (up2 - up1).unit();
  const auto outgoing = (down2 - down1).unit();
  // Projected scattering angles are useful diagnostic variables; theta is
  // their small-angle two-dimensional magnitude and is later used as θ²
  // voxel weight in the Python reconstruction.
  const auto thetaX = std::atan2(outgoing.x(), outgoing.z()) -
                      std::atan2(incoming.x(), incoming.z());
  const auto thetaY = std::atan2(outgoing.y(), outgoing.z()) -
                      std::atan2(incoming.y(), incoming.z());
  const auto theta = std::sqrt(thetaX * thetaX + thetaY * thetaY);

  // PoCA (Point of Closest Approach):
  // find the closest point on each fitted line, then use their midpoint as an
  // approximate location of the dominant scattering. Nearly parallel tracks
  // make the denominator close to zero; in that rare case s=t=0 is retained
  // rather than dividing by a small value.
  const auto u = incoming;
  const auto v = outgoing;
  const auto w0 = up1 - down1;
  const auto a = u.dot(u);
  const auto b = u.dot(v);
  const auto c = v.dot(v);
  const auto d = u.dot(w0);
  const auto e = v.dot(w0);
  const auto denominator = a * c - b * b;

  G4double s = 0.;
  G4double t = 0.;
  if (std::abs(denominator) > 1.e-12) {
    s = (b * e - c * d) / denominator;
    t = (a * e - b * d) / denominator;
  }
  const auto pocaIn = up1 + s * u;
  const auto pocaOut = down1 + t * v;
  const auto poca = 0.5 * (pocaIn + pocaOut);

  // Geant4's internal length unit is converted to mm here, and angular values
  // to mrad, so the CSV can be inspected directly in the Python notebook.
  output << event->GetEventID();
  for (int i = 0; i < 4; ++i) {
    output << ',' << fX[i] / mm << ',' << fY[i] / mm << ',' << fZ[i] / mm;
  }
  output << ',' << poca.x() / mm << ',' << poca.y() / mm << ',' << poca.z() / mm
         << ',' << thetaX / mrad << ',' << thetaY / mrad << ',' << theta / mrad
         << '\n';
  fRunAction->AddTrack();
}

}
