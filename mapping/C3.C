#include "PHASIC++/Channels/Single_Channel.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/MyStrStream.H"
#include "ATOOLS/Math/Poincare.H"
#include "ATOOLS/Math/Random.H"
#include "PHASIC++/Channels/Channel_Elements.H"
#include "PHASIC++/Channels/Vegas.H"

// #define LOG_SAMPLING

using namespace PHASIC;
using namespace ATOOLS;

namespace PHASIC {
  class C3 : public Single_Channel {
    Vegas* p_vegas;
    double alpha;
    int channel;
  public:
    C3(int,int,Flavour*,Integration_Info * const);
    ~C3();
    void   GenerateWeight(Vec4D *,Cut_Data *);
    void   GeneratePoint(Vec4D *,Cut_Data *,double *);
    void   AddPoint(double);
    void   MPISync()                 { p_vegas->MPISync(); }
    void   Optimize()                { p_vegas->Optimize(); } 
    void   EndOptimize()             { p_vegas->EndOptimize(); } 
    void   WriteOut(std::string pId) { p_vegas->WriteOut(pId); } 
    void   ReadIn(std::string pId)   { p_vegas->ReadIn(pId); } 
    void   ISRInfo(int &,double &,double &);
    std::string ChID();
    
double MasslessPoint(const double &smin,const double &smax,const double ran)
{
  DEBUG_FUNC("");
#ifdef LOG_SAMPLING
  double s = smin*pow(smax/smin,ran);
#else
  double s = pow(pow(smin,1.-alpha)*(1.-ran)+pow(smax,1.-alpha)*ran,1./(1.-alpha));
#endif
  DEBUG_VAR(s<<" "<<smin<<" "<<smax<<" "<<ran);
  return s;
}

double MasslessWeight(const double &smin,const double &smax,const Vec4D &p,double &ran)
{
  DEBUG_FUNC("");
  double s = p.Abs2();
#ifdef LOG_SAMPLING
  double I = log(smax/smin);
  ran = log(s/smin)/I;
  double wgt = s*I/(2.*M_PI);
#else
  double I = (pow(smax,1.-alpha)-pow(smin,1.-alpha))/(1.-alpha);
  ran = (pow(s,1.-alpha)-pow(smin,1.-alpha))/(1.-alpha)/I;
  double wgt = pow(s,alpha)*I/(2.*M_PI);
#endif
  DEBUG_VAR(s<<" "<<smin<<" "<<smax<<" "<<ran);
  return wgt;
}

  };
}

extern "C" Single_Channel * Getter_C3(int nin,int nout,Flavour* fl,Integration_Info * const info) {
  return new C3(nin,nout,fl,info);
}

double Lambda(const double &a,const double &b,const double &c) { return sqr(a-b-c)-4*b*c; }

const double eps=1.e-6;

std::pair<Vec4D,Vec4D> IsoPoint(const Vec4D &p,const double &s1,const double &s2,const double &alpha,const double *rans)
{
  DEBUG_FUNC("");
  double ecm = p.Mass();
  // double ct = 2.*rans[0]-1.;
  double ct = 1.-pow(pow(eps,1.-alpha)*(1.-rans[0])+pow(2.-eps,1.-alpha)*rans[0],1./(1.-alpha));
  double st = sqrt(1.-ct*ct);
  double phi = 2.*M_PI*rans[1];
  Vec4D pl(0.,Vec3D(p)/p.P());
  if (IsZero(p.PSpat2())) pl=Vec4D(0.,0.,0.,1.);
  Vec4D pt1(0.,cross(Vec3D(pl),Vec3D(1.,0.,0.)));
  pt1=pt1/pt1.P();
  Vec4D pt2(0.,cross(Vec3D(pt1),Vec3D(pl)));
  pt2=pt2/pt2.P();
  double ps = sqrt(Lambda(p.Abs2(),s1,s2))/(2.*ecm);
  Vec4D p1 = ps*(st*(cos(phi)*pt1+sin(phi)*pt2)+ct*pl);
  p1[0] = (p.Abs2()+s1-s2)/(2.*ecm);
  Vec4D p2(ecm-p1[0],-p1[1],-p1[2],-p1[3]);
  Poincare cms(p);
  DEBUG_VAR(ct<<" "<<phi);
  DEBUG_VAR(p1);
  DEBUG_VAR(p2);
  cms.BoostBack(p1);
  cms.BoostBack(p2);
  DEBUG_VAR(p1);
  DEBUG_VAR(p2);
  DEBUG_VAR(rans[0]<<" "<<rans[1]);
  return std::make_pair(p1,p2);
}

double IsoWeight(const Vec4D &p,const Vec4D &p1,const Vec4D &p2,const double &alpha,double *rans)
{
  DEBUG_FUNC("");
  double ecm = p.Mass();
  Poincare cms(p);
  Vec4D q1(p1), q2(p2);
  cms.Boost(q1);
  cms.Boost(q2);
  Vec4D pl(0.,Vec3D(p)/p.P());
  if (IsZero(p.PSpat2())) pl=Vec4D(0.,0.,0.,1.);
  Vec4D pt1(0.,cross(Vec3D(pl),Vec3D(1.,0.,0.)));
  pt1=pt1/pt1.P();
  Vec4D pt2(0.,cross(Vec3D(pt1),Vec3D(pl)));
  pt2=pt2/pt2.P();
  double ct = -(pl*q1)/q1.PSpat();
  double phi = atan((q1*pt2)/(q1*pt1));
  if ((q1*pt1)>0) phi += M_PI;
  else if (phi<0) phi += 2.*M_PI;
  // rans[0] = (1.+ct)/2.;
  double I = (pow(2.-eps,1.-alpha)-pow(eps,1.-alpha))/(1.-alpha);
  rans[0] = (pow(1.-ct,1.-alpha)-pow(eps,1.-alpha))/(1.-alpha)/I;
  double ctw = pow(1.-ct,alpha)*I/2.;
  rans[1] = phi/(2.*M_PI);
  DEBUG_VAR(ct<<" "<<phi);
  DEBUG_VAR(p1);
  DEBUG_VAR(p2);
  DEBUG_VAR(q1);
  DEBUG_VAR(q2);
  DEBUG_VAR(rans[0]<<" "<<rans[1]);
  double ps = sqrt(Lambda(p.Abs2(),p1.Abs2(),p2.Abs2()))/(2.*ecm);
  double wgt = 4.*M_PI*ctw*ps/(16.*sqr(M_PI)*ecm);
  return wgt;
}
 
void C3::GeneratePoint(Vec4D * p,Cut_Data * cuts,double * _ran)
{
  DEBUG_FUNC("");
  double *ran = p_vegas->GeneratePoint(_ran);
  for(int i=0;i<rannum;i++) rans[i]=ran[i];
  double shat((p[0]+p[1]).Abs2());
  double s13 = MasslessPoint(1.e-3,shat,rans[0]);
  std::pair<Vec4D,Vec4D> p13_2 = IsoPoint(p[0]+p[1],s13,0.,0.,&rans[1]);
  std::pair<Vec4D,Vec4D> p1_3 = IsoPoint(p13_2.first,0.,0.,.999,&rans[3]);
  p[2] = p1_3.second;
  channel = ATOOLS::ran->Get()>0.5?0:1; 
  if (channel) {
    DEBUG_VAR("channel 1");
    p[3] = p1_3.first;
    p[4] = p13_2.second;
  }
  else {
    DEBUG_VAR("channel 2");
    p[4] = p1_3.first;
    p[3] = p13_2.second;
  }
  DEBUG_VAR(p[2]<<" "<<p[2].Abs2());
  DEBUG_VAR(p[3]<<" "<<p[3].Abs2());
  DEBUG_VAR(p[4]<<" "<<p[4].Abs2());
  DEBUG_VAR((p[2]+p[3]+p[4])<<" "<<(p[2]+p[3]+p[4]).Abs2());
  DEBUG_VAR(rans[0]<<" "<<rans[1]<<" "<<rans[2]<<" "<<rans[3]<<" "<<rans[4]);
}

void C3::GenerateWeight(Vec4D *p,Cut_Data *cuts)
{
  double rans1[5], rans2[5];
  double shat((p[0]+p[1]).Abs2());
  double ws13 = MasslessWeight(1.e-3,shat,p[3]+p[2],rans1[0]);
  double wp13_2 = IsoWeight(p[0]+p[1],p[3]+p[2],p[4],0.,&rans1[1]);
  double wp1_3 = IsoWeight(p[3]+p[2],p[3],p[2],.999,&rans1[3]);
  double vw1 = p_vegas->GenerateWeight(rans);
  DEBUG_VAR(ws13<<" "<<wp13_2<<" "<<wp1_3<<" "<<vw1);
  double ws23 = MasslessWeight(1.e-3,shat,p[4]+p[2],rans2[0]);
  double wp23_1 = IsoWeight(p[0]+p[1],p[4]+p[2],p[3],0.,&rans2[1]);
  double wp2_3 = IsoWeight(p[4]+p[2],p[4],p[2],.999,&rans2[3]);
  double vw2 = p_vegas->GenerateWeight(rans2);
  DEBUG_VAR(ws23<<" "<<wp23_1<<" "<<wp2_3<<" "<<vw2);
  weight = 1./(0.5/(wp13_2*ws13*wp1_3*vw1)+0.5/(wp23_1*ws23*wp2_3*vw2));
  for (int i(0);i<5;++i) {
    if (!IsEqual(rans[i],channel?rans1[i]:rans2[i],1.0e-6))
      msg_Error()<<METHOD<<"(): Inconsistent ran "<<i
		 <<": "<<rans[i]<<" vs. "<<(channel?rans1[i]:rans2[i])<<std::endl;
  }
}

C3::C3(int nin,int nout,Flavour* fl,Integration_Info * const info)
       : Single_Channel(nin,nout,fl)
{
  name = std::string("C3");
  rannum = 5;
  rans  = new double[rannum];
  p_vegas = new Vegas(rannum,100,name);
  alpha = ToType<double>(rpa->gen.Variable("AMEGIC_SCHANNEL_ALPHA"));
}

C3::~C3()
{
  delete p_vegas;
}

void C3::ISRInfo(int & type,double & mass,double & width)
{
  type  = 2;
  mass  = 0;
  width = 0.;
}

void C3::AddPoint(double Value)
{
  Single_Channel::AddPoint(Value);
  p_vegas->AddPoint(Value,rans);
}
std::string C3::ChID()
{
  return std::string("XXXCGND$I_2_3$I_4_23$MTH_23$ZS_0$");
}
