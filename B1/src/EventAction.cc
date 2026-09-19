#include "EventAction.hh"
#include "RunAction.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include <cmath>
#include <fstream>

namespace B1
{

namespace
{
// The two incoming and two outgoing detector planes.
constexpr int kUpstream1 = 0;
constexpr int kUpstream2 = 1;
constexpr int kDownstream1 = 2;
constexpr int kDownstream2 = 3;

}

EventAction::EventAction(RunAction* runAction) : fRunAction(runAction) {}

void EventAction::BeginOfEventAction(const G4Event*)
{
  for (int i = 0; i < 4; ++i) fHasHit[i] = false;
}

void EventAction::SetHit(const G4String& detector, G4double x, G4double y,
                         G4double z)
{
  int index = -1;
  if (detector == "UpstreamDetector1") index = kUpstream1;
  if (detector == "UpstreamDetector2") index = kUpstream2;
  if (detector == "DownstreamDetector1") index = kDownstream1;
  if (detector == "DownstreamDetector2") index = kDownstream2;
  if (index < 0 || fHasHit[index]) return;

  fHasHit[index] = true;
  fX[index] = x;
  fY[index] = y;
  fZ[index] = z;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  static std::ofstream output("../../muon_analysis/muon_tracks.csv");
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

  for (int i = 0; i < 4; ++i) {
    if (!fHasHit[i]) return;
  }

  const G4ThreeVector up1(fX[kUpstream1], fY[kUpstream1], fZ[kUpstream1]);
  const G4ThreeVector up2(fX[kUpstream2], fY[kUpstream2], fZ[kUpstream2]);
  const G4ThreeVector down1(fX[kDownstream1], fY[kDownstream1], fZ[kDownstream1]);
  const G4ThreeVector down2(fX[kDownstream2], fY[kDownstream2], fZ[kDownstream2]);

  // Direction vectors of the fitted incoming and outgoing straight tracks.
  const auto incoming = (up2 - up1).unit();
  const auto outgoing = (down2 - down1).unit();
  const auto thetaX = std::atan2(outgoing.x(), outgoing.z()) -
                      std::atan2(incoming.x(), incoming.z());
  const auto thetaY = std::atan2(outgoing.y(), outgoing.z()) -
                      std::atan2(incoming.y(), incoming.z());
  const auto theta = std::sqrt(thetaX * thetaX + thetaY * thetaY);

  // PoCA: midpoint of the closest approach points of the two fitted lines.
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
