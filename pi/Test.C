#include "Foam.H"

#include "Tools.H"
#include <termios.h>
#include <unistd.h>

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
    return weight;
  }
};// end of class Circle

int main(int argc,char **argv)
{
  PRINT_INFO("Initialize integrator");
  Foam integrator;
  // set dimension
  integrator.SetDimension(2);
  // set max cell number
  integrator.SetNCells(10000);
  // set point number between optimization steps
  integrator.SetNOpt(1000);
  // set max point number
  integrator.SetNMax(2000000);
  // set error
  integrator.SetError(5.0e-4);
  // integrate
  integrator.Initialize();
  int type(0);
  if (argc>1) {
    if (strcmp(argv[1],"camel")==0) type=0;
    if (strcmp(argv[1],"line")==0) type=1;
    if (strcmp(argv[1],"circle")==0) type=2;
  }
  if (type==0) {
    PRINT_INFO("Integrate camel");
    Camel camel;
    integrator.Integrate(&camel);
  }
  if (type==1) {
    PRINT_INFO("Integrate line");
    Line line;
    integrator.Integrate(&line);
  }
  if (type==2) {
    PRINT_INFO("Integrate circle");
    Circle circle;
    integrator.Integrate(&circle);
  }
  return 0;
}
