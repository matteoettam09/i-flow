#include "Foam.H"

#include "Tools.H"
#include <termios.h>
#include <unistd.h>
#ifdef USING__ROOT
#include "TApplication.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH2D.h"
TApplication *s_root(NULL);
std::vector<std::pair<std::string,TObject*> > s_objects;
class PS_Histogram: public TH2D {
private:
  TH2D *p_ps;
public:
  // constructor
  PS_Histogram(const char *name,const char *title,
	       const size_t nbinsx,const double xmin,const double xmax,
	       const size_t nbinsy,const double ymin,const double ymax):
    TH2D(name,title,nbinsx,xmin,xmax,nbinsy,ymin,ymax),
    p_ps(new TH2D((std::string(name)+"_ps").c_str(),
		  (std::string(title)+"_ps").c_str(),
		  nbinsx,xmin,xmax,nbinsy,ymin,ymax)) {}
  // destructor
  ~PS_Histogram() {}
  // member functions
  Int_t Fill(const Double_t x,const Double_t y,const Double_t weight);
  void Draw(Option_t *option="");
};// end of class PS_Histogram
Int_t PS_Histogram::Fill(const Double_t x,const Double_t y,
			 const Double_t weight)
{
  p_ps->Fill(x,y,1.0);
  return TH2D::Fill(x,y,weight);
}
void PS_Histogram::Draw(Option_t *option)
{
  for (Int_t i=0;i<GetNbinsX();++i)
    for (Int_t j=0;j<GetNbinsY();++j)
      SetBinContent(i,j,p_ps->GetBinContent(i,j)==0.0?0.0:
		    GetBinContent(i,j)/
		    p_ps->GetBinContent(i,j));
  TVirtualPad *psc=gPad;
  Int_t logx=psc->GetLogx();
  Int_t logy=psc->GetLogy();
  Int_t logz=psc->GetLogz();
  psc->Divide(2,1);
  psc->cd(1);
  gPad->SetLogx(logx);
  gPad->SetLogy(logy);
  gPad->SetLogz(logz);
  TH2D::Draw(option);
  psc->cd(2);
  gPad->SetLogx(logx);
  gPad->SetLogy(logy);
  gPad->SetLogz(logz);
  p_ps->Draw(option);
}
#endif

using namespace FOAM;

class Camel: public Foam_Integrand {
public:
  bool m_fill;
  Camel(): m_fill(false) {}
  double operator()(const std::vector<double> &point)
  {
    const double dx1=0.25, dy1=0.25, w1=1./0.004;
    const double dx2=0.75, dy2=0.75, w2=1./0.004;
    double weight=exp(-w1*((point[0]-dx1)*(point[0]-dx1)+
			   (point[1]-dy1)*(point[1]-dy1)))+
      exp(-w2*((point[0]-dx2)*(point[0]-dx2)+
	       (point[1]-dy2)*(point[1]-dy2)));
#ifdef USING__ROOT
    static PS_Histogram *ps=NULL;
    if (ps==NULL) {
      ps = new PS_Histogram("camel","camel",100,0.,1.,100,0.,1.);
      s_objects.push_back(std::pair<std::string,TObject*>("camel",ps));
    }
    if (m_fill) ps->Fill(point[0],point[1],weight);
#endif
    return weight;
  }
};// end of class Camel

class Line: public Foam_Integrand {
public:
  bool m_fill;
  Line(): m_fill(false) {}
  double operator()(const std::vector<double> &point)
  {
    const double w1=1./0.004, mm=0.01;
    double weight=exp(-w1*sqr(point[1]+point[0]-1.0));
    if (point[0]<mm || point[0]>1.0-mm ||
 	point[1]<mm || point[1]>1.0-mm) weight=0.0;
#ifdef USING__ROOT
    static PS_Histogram *ps=NULL;
    if (ps==NULL) {
      ps = new PS_Histogram("line","line",100,0.,1.,100,0.,1.);
      s_objects.push_back(std::pair<std::string,TObject*>("line",ps));
    }
    if (m_fill) ps->Fill(point[0],point[1],weight);
#endif
    return weight;
  }
};// end of class Line

class Circle: public Foam_Integrand {
public:
  bool m_fill;
  Circle(): m_fill(false) {}
  double operator()(const std::vector<double> &point)
  {
    const double dx1=0.4, dy1=0.6, rr=0.25, w1=1./0.004, ee=3.0;
    double weight=pow(point[1],ee)*
      exp(-w1*dabs(sqr(point[1]-dy1)+
		   sqr(point[0]-dx1)-sqr(rr)));
    weight+=pow(1.0-point[1],ee)*
      exp(-w1*dabs(sqr(point[1]-1.0+dy1)+
		   sqr(point[0]-1.0+dx1)-sqr(rr)));
#ifdef USING__ROOT
    static PS_Histogram *ps=NULL;
    if (ps==NULL) {
      ps = new PS_Histogram("circle","circle",100,0.,1.,100,0.,1.);
      s_objects.push_back(std::pair<std::string,TObject*>("circle",ps));
    }
    if (m_fill) ps->Fill(point[0],point[1],weight);
#endif
    return weight;
  }
};// end of class Circle

int main(int argc,char **argv)
{
#ifdef USING__ROOT
  int argcf=1;
  char **argvf = new char*[1];
  argvf[0] = new char[5];
  strcpy(argvf[0],"root");
  s_root = new TApplication("MyRoot",&argcf,argvf);
  gStyle->SetPalette(1);
#endif
  PRINT_INFO("Initialize integrator");
  Foam integrator;
  // set dimension
  integrator.SetDimension(2);
  // set max cell number
  integrator.SetNCells(500);
  // set point number between optimization steps
  integrator.SetNOpt(1000);
  // set max point number
  integrator.SetNMax(2000000);
  // set error
  integrator.SetError(5.0e-4);
  // integrate
  integrator.Initialize();
  PRINT_INFO("Integrate camel");
  Camel camel;
  integrator.Integrate(&camel);
#ifdef USING__ROOT
  camel.m_fill=true;
  for (size_t i(0);i<1000000;++i) integrator.Point();
#endif
  /*
  integrator.Initialize();
  PRINT_INFO("Integrate line");
  Line line;
  integrator.Integrate(&line);
#ifdef USING__ROOT
  line.m_fill=true;
  for (size_t i(0);i<1000000;++i) integrator.Point();
#endif
  integrator.Initialize();
  PRINT_INFO("Integrate circle");
  Circle circle;
  integrator.Integrate(&circle);
#ifdef USING__ROOT
  circle.m_fill=true;
  for (size_t i(0);i<1000000;++i) integrator.Point();
#endif
  */
#ifdef USING__ROOT
  for (size_t i(0);i<s_objects.size();++i) {
    new TCanvas((s_objects[i].first+"_c").c_str(),
		(s_objects[i].first+"_c").c_str());
    s_objects[i].second->Draw("lego2");
  }
  if (s_objects.size()>0) s_root->Run(kTRUE);
  delete s_root;
#endif
  return 0;
}
