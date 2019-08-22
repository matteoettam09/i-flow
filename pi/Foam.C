#include "Foam.H"

#include "Random.H"
#ifndef USING__PI_only
#include "Exception.H"
#include "Run_Parameter.H"
#else
#include "Tools.H"
#endif
#include <iomanip>
#include <unistd.h>
#include <algorithm>

#define USING__IMMEDIATE_DELETE

using namespace FOAM;

class Order_X {
private:
  size_t m_dim;
public:
  // constructor
  inline Order_X(const size_t &dim): m_dim(dim) {}
  // inline functions
  inline bool operator()(const std::pair<std::vector<double>,double> &a,
			 const std::pair<std::vector<double>,double> &b)
  { return a.first[m_dim]<b.first[m_dim]; }
};// end of class Order_X

long unsigned int Foam_Channel::s_npoints(0);
long unsigned int s_nmaxpoints(1000000);

Foam_Integrand::~Foam_Integrand()
{
}

double Foam_Integrand::operator()(const std::vector<double> &point)
{
  std::cerr<<"Error: operator() undefined"<<std::endl;
  abort();
}

void Foam_Integrand::AddPoint(const double value,const double weight,
				   const int mode)
{
}

void Foam_Integrand::FinishConstruction(const double apweight)
{
}

#define DFORMAT std::setw(15)

std::ostream &FOAM::operator<<(std::ostream &str,
				 const Foam_Channel &channel)
{
  str<<"("<<&channel<<"): {\n"
     <<"   m_alpha  = "<<DFORMAT<<channel.Alpha()
     <<" <- oldalpha   = "<<DFORMAT<<channel.OldAlpha()<<"\n"
     <<"   m_weight = "<<DFORMAT<<channel.Weight()
     <<"    m_max      = "<<DFORMAT<<channel.Max()<<"\n"
     <<"   m_sum    = "<<DFORMAT<<channel.Sum()
     <<" -> integral   = "<<DFORMAT<<channel.Mean()<<"\n"
     <<"   m_sum2   = "<<DFORMAT<<channel.Sum2()
     <<" -> error      = "<<DFORMAT<<channel.Sigma()<<"\n"
     <<"   m_np     = "<<DFORMAT<<channel.Points()
     <<" -> rel. error = "<<DFORMAT<<channel.Sigma()/channel.Mean()<<"\n"
     <<"   m_ssum   = "<<DFORMAT<<channel.SSum()
     <<" -> integral   = "<<DFORMAT<<channel.SMean()<<"\n"
     <<"   m_ssum2  = "<<DFORMAT<<channel.SSum2()
     <<" -> error      = "<<DFORMAT<<channel.SSigma()<<"\n"
     <<"   m_snp    = "<<DFORMAT<<channel.SPoints()
     <<" -> rel. error = "<<DFORMAT<<channel.SSigma()/channel.SMean()<<"\n"
     <<"   m_this   = ";
  for (size_t i(0);i<channel.m_this.size();++i) 
    str<<DFORMAT<<channel.m_this[i]<<" ";
  str<<"\n   m_next   = ";
  for (size_t i(0);i<channel.m_next.size();++i) 
    str<<DFORMAT<<channel.m_next[i]<<" ";
  str<<"\n              ";
  for (size_t i(0);i<channel.m_next.size();++i) 
    str<<DFORMAT<<(channel.m_next[i]!=NULL?
		   channel.m_next[i]->m_this[i]:0.0)<<" ";
  return str<<"\n}"<<std::endl;
}

Foam_Channel::Foam_Channel(Foam *const integrator):
  p_integrator(integrator),
  m_alpha(0.0), m_oldalpha(0.0), m_weight(0.0), m_loss(0.0),
  m_sum(0.0), m_sum2(0.0), m_max(0.0), m_np(0.0), 
  m_ssum(0.0), m_ssum2(0.0), m_snp(0.0),
  m_split(-1) {}

Foam_Channel::
Foam_Channel(Foam *const integrator,
		  Foam_Channel *const prev,const size_t i,
		  const double &pos):
  p_integrator(integrator),
  m_alpha(0.0), m_oldalpha(0.0), m_weight(0.0), m_loss(0.0),
  m_sum(0.0), m_sum2(0.0), m_max(0.0), m_np(0.0), 
  m_ssum(0.0), m_ssum2(0.0), m_snp(0.0),
  m_this(prev->m_this),
  m_next(prev->m_next),
  m_split(-1)
{
  if (prev->Boundary()) THROW(fatal_error,"Attempt to split boundary cell.");
  if (i>=prev->m_this.size()) THROW(fatal_error,"Inconsistent dimensions.");
  if (pos!=std::numeric_limits<double>::max()) m_this[i]=pos;
  else m_this[i]=(m_next[i]->m_this[i]+m_this[i])/2.0;
  m_next[i]=prev->m_next[i];
  prev->m_next[i]=this;
  prev->SetWeight();
  SetWeight();
  m_alpha=(prev->m_alpha/=2.0);
  if (!prev->m_points.empty()) {
    std::sort(prev->m_points.begin(),prev->m_points.end(),Order_X(i));
    std::vector<std::pair<std::vector<double>,double> >::iterator
      jit(prev->m_points.begin());
    for (size_t j(0);j<prev->m_points.size();++j,++jit) {
      if (jit->first[i]>=m_this[i]) {
	m_points.insert(m_points.begin(),jit,prev->m_points.end());
	prev->m_points.erase(jit,prev->m_points.end());
	break;
      }
    }
  }
  prev->Reset();
  Reset();
  msg_Debugging()<<METHOD<<"("<<prev<<","<<i<<"): {\n"<<*prev<<*this<<"}\n";
  if (m_weight<=0.0 || prev->m_weight<=0.0) 
    THROW(fatal_error,"New cell has nonpositive weight.");
}
    
Foam_Channel::~Foam_Channel()
{
}

double Foam_Channel::Point(Foam_Integrand *const function,
				std::vector<double> &point)
{
  if (point.size()!=m_this.size())
    THROW(fatal_error,"Inconsistent dimensions.");
  if (Boundary()) THROW(fatal_error,"Boundary cell selected.");
  for (size_t i(0);i<m_this.size();++i) 
    point[i]=m_this[i]+ran.Get()*(m_next[i]->m_this[i]-m_this[i]);
  double cur((*function)(point)), weight(cur*m_weight);
  if (!(cur<0.0) && !(cur>=0.0)) THROW(critical_error,"Integrand is nan.");
  ++m_np;
  m_sum+=weight;
  m_sum2+=sqr(weight);
  m_max=FOAM::Max(m_max,dabs(weight));
  if (p_integrator->RunMode()==rmc::construct) {
    m_points.push_back(std::pair<std::vector<double>,double>(point,cur));
    if (++s_npoints>s_nmaxpoints) THROW(fatal_error,"Too many stored points.");
  }
  else {
    function->AddPoint(cur,m_weight/m_alpha);
  }
  return cur;
}

bool Foam_Channel::Find(const std::vector<double> &point) const
{
  if (point.size()!=m_this.size()) 
    THROW(fatal_error,"Inconsistent dimesions.");
  for (size_t i(0);i<m_this.size();++i) 
    if (point[i]<m_this[i] || 
	(m_next[i]!=NULL && point[i]>=m_next[i]->m_this[i])) return false;
  return true;
}

void Foam_Channel::Reset()
{
  m_sum=m_sum2=m_max=m_np=0.0;
  if (!m_points.empty()) {
    for (std::vector<std::pair<std::vector<double>,double> >::const_iterator
	   pit(m_points.begin());pit!=m_points.end();++pit) {
      ++m_np;
      m_sum+=pit->second;
      m_sum2+=sqr(pit->second);
      m_max=FOAM::Max(m_max,dabs(pit->second));
    }
    m_sum*=m_weight;
    m_sum2*=sqr(m_weight);
    m_max*=m_weight;
  }
}

void Foam_Channel::DeletePoints(const int mode) 
{ 
  if (m_points.empty()) return;
  msg_Tracking()<<METHOD<<"(): Delete points in channel "<<this<<".\n";
  s_npoints-=m_points.size();
  if (mode==1) 
    for (size_t i(0);i<m_points.size();++i)
      p_integrator->Function()->
	AddPoint(m_points[i].second,m_weight/m_alpha,1);
  m_points.clear(); 
}

void Foam_Channel::SetWeight()
{
  if (Boundary()) return;
  m_weight=1.0;
  for (size_t i(0);i<m_this.size();++i) 
    m_weight*=m_next[i]->m_this[i]-m_this[i];
}

void Foam_Channel::SetAlpha(const double &alpha)
{ 
  if (Boundary()) return;
  m_alpha=alpha;
}

void Foam_Channel::Store()
{
  m_snp+=m_np;
  m_ssum+=m_sum/m_alpha;
  m_ssum2+=m_sum2/sqr(m_alpha);
}

void Foam_Channel::SelectSplitDimension(const std::vector<int> &nosplit)
{
  if (m_points.empty()) {
    if (m_split<0) THROW(fatal_error,"No phase space points.");
    return;
  }
  double diff(0.0);
  for (size_t dim(0);dim<m_this.size();++dim) {
    if (nosplit.at(dim)) continue;
    std::sort(m_points.begin(),m_points.end(),Order_X(dim));
    std::pair<double,double> s2=
      p_integrator->Loss(this,m_points.size()/2);
    double sl(0.0), sr(m_sum);
    for (size_t i(0);i<m_points.size()/2;++i) {
      sl+=dabs(m_points[i].second*m_weight);
      sr-=dabs(m_points[i].second*m_weight);
    }
    double varl(s2.first/sl), varr(s2.second/sr);
    if (dabs(varl-varr)>diff) {
      diff=dabs(varl-varr);
      m_split=dim;
    }
  }
  m_loss=p_integrator->Loss(this).first;
#ifdef USING__IMMEDIATE_DELETE
  DeletePoints();
#endif
}

bool Foam_Channel::
WriteOut(std::fstream *const file,
	 std::map<Foam_Channel*,size_t> &pmap) const
{
  (*file)<<"[ "<<m_alpha<<" "<<m_oldalpha<<" "<<m_sum<<" "
	 <<m_sum2<<" "<<m_max<<" "<<m_np<<" "<<m_ssum
	 <<" "<<m_ssum2<<" "<<m_snp<<" "<<m_loss<<" ( ";
  for (size_t i(0);i<m_this.size();++i) (*file)<<m_this[i]<<" ";
  (*file)<<") ( ";
  for (size_t i(0);i<m_next.size();++i) (*file)<<pmap[m_next[i]]<<" ";
  (*file)<<") ]"<<std::endl;
  return true;
}

bool Foam_Channel::ReadIn(std::fstream *const file,
			       std::map<size_t,Foam_Channel*> &pmap)
{
  if (file->eof()) return false;
  std::string dummy;
  (*file)>>dummy>>m_alpha>>m_oldalpha>>m_sum>>m_sum2
	 >>m_max>>m_np>>m_ssum>>m_ssum2>>m_snp>>m_loss>>dummy;
  for (size_t i(0);i<m_this.size();++i) {
    if (file->eof()) return false;
    (*file)>>m_this[i];
  }
  (*file)>>dummy>>dummy;
  for (size_t i(0);i<m_next.size();++i) {
    if (file->eof()) return false;
    size_t pos;
    (*file)>>pos;
    m_next[i]=(Foam_Channel*)pmap[pos];
  }
  (*file)>>dummy>>dummy;
  if (file->eof()) return false;
  return true;
}

void Foam_Channel::CreateRoot(Foam *const integrator,
				   const std::vector<double> &min,
				   const std::vector<double> &max,
				   std::vector<Foam_Channel*> &channels)
{
  if (min.size()!=max.size()) THROW(fatal_error,"Inconsistent dimensions.");
  if (!channels.empty()) THROW(critical_error,"Initialized channels found.");
  Foam_Channel *root(new Foam_Channel(integrator));
  channels.push_back(root);
  root->m_this=min;
  root->m_next.resize(max.size());
  for (size_t i(0);i<max.size();++i) {
    Foam_Channel *next(new Foam_Channel(integrator));
    channels.push_back(next);
    next->m_next.resize(max.size(),NULL);
    next->m_this=min;
    next->m_this[i]=max[i];
    root->m_next[i]=next;
  }
  root->SetAlpha(1.0);
  root->SaveAlpha();
  root->SetWeight();
}

Foam::Foam():
  m_nopt(10000), m_nmax(1000000), m_error(0.01), m_scale (1.0),
  m_apweight(1.0), m_sum(0.0), m_sum2(0.0), m_max(0.0), 
  m_np(0.0), m_nrealp(0.0), 
  m_smax(std::deque<double>(3,0.0)), 
  m_ncells(1000), m_split(1), m_shuffle(1), m_last(0), 
  m_mode(0),
  m_rmode(rmc::none),
  m_vname("I") {}

Foam::~Foam()
{
  while (!m_channels.empty()) {
    delete m_channels.back();
    m_channels.pop_back();
  }
}

void Foam::SetDimension(const size_t dim)
{
  m_rmin.resize(dim,0.0);
  m_rmax.resize(dim,1.0);
  m_nosplit.resize(dim);
  for (size_t i(0);i<dim;++i) m_nosplit[i]=false;
}

void Foam::Reset()
{
  m_sum=m_sum2=m_max=m_np=m_nrealp=0.0; 
  while (!m_channels.empty()) {
    delete m_channels.back();
    m_channels.pop_back();
  }
  m_asum.clear();
}

void Foam::Initialize()
{
  Reset();
  if (m_rmin.empty() || m_rmax.empty())
    THROW(fatal_error,"Zero dimensional integral request.");
  if (m_rmin.size()!=m_rmax.size())
    THROW(fatal_error,"Inconsistent dimensions.");
  m_point.resize(m_rmin.size());
  Foam_Channel::CreateRoot(this,m_rmin,m_rmax,m_channels);
  m_ndiced=0;
}

class Python_Function: public Foam_Integrand {
public:
  PyObject *self, *method;
  Python_Function(PyObject *_self):
    self(_self), method(PyBytes_FromString("__call__")) {}
  PyObject *vectorToList_Double(const std::vector<double> &data)
  {
    PyObject* listObj = PyList_New( data.size() );
    if (!listObj) throw std::logic_error("Out of memory");
    for (unsigned int i = 0; i < data.size(); i++) {
      PyObject *num = PyFloat_FromDouble(data[i]);
      if (!num) { Py_DECREF(listObj); throw std::logic_error("Out of memory"); }
      PyList_SET_ITEM(listObj,i,num);
    }
    return listObj;
  }
  double operator()(const std::vector<double> &x)
  {
    PyObject *input(vectorToList_Double(x));
    PyObject *output(PyObject_CallMethodObjArgs(self,method,input,NULL));
    double res = PyFloat_AsDouble(output);
    Py_DECREF(input);
    Py_DECREF(output);
    return res;
  }
};

double Foam::Integrate(PyObject *const function)
{
  Python_Function *pyfunc = new Python_Function(function);
  return Integrate(pyfunc);
}

double Foam::Integrate(Foam_Integrand *const function)
{
  if (m_channels.empty()) Initialize();
  p_function=function;
  msg_Debugging()<<METHOD<<"("<<function<<"): {\n";
  for (size_t i(0);i<m_channels.size();++i) msg_Debugging()<<*m_channels[i];
  msg_Debugging()<<"}"<<std::endl;
  m_apweight=m_channels[0]->Alpha();
  m_rmode=rmc::construct;
  long unsigned int nfirst((m_channels.size()-m_point.size())*m_nopt/2);
  for (long unsigned int n(0);n<nfirst;++n) Point();
  Split();
#ifndef USING__PI_only
  msg_Info()<<tm::curoff;
#endif
  while (((long unsigned int)m_np)<m_nmax/2 &&
	 m_channels.size()-m_point.size()<m_ncells) {
    for (;m_ndiced<m_nopt;++m_ndiced) Point();
    CheckTime();
    if (Update(0)<m_error) break;
    Split();
  }
  for (;m_ndiced<m_nopt;++m_ndiced) Point();
  Update(0);
  msg_Info()<<mm_down(1)<<std::endl;
  p_function->FinishConstruction(m_apweight);
  double asum(0.0);
  m_asum.resize(m_channels.size()+1,0.0);
  for (size_t i(0);i<m_channels.size();++i) {
    if (!m_channels[i]->Boundary()) {
      m_channels[i]->SetAlpha(m_apweight);
      m_channels[i]->DeletePoints(1);
      m_channels[i]->Store();
      m_channels[i]->Reset();
      asum+=m_apweight;
    }
    m_asum[i+1]=asum;
  }
  m_rmode=rmc::shuffle;
  size_t add(0);
  long unsigned int nsopt(m_point.size()*m_nopt);
  while (((long unsigned int)m_np)<m_nmax) {
    for (long unsigned int n(0);n<nsopt;++n) Point();
    CheckTime();
    if (Update(1)<m_error && 
	add++>=FOAM::Max((size_t)5,m_point.size())) break;
    Shuffle();
  }
  msg_Info()<<mm_down(1)<<std::endl;
  m_rmode=rmc::run;
#ifndef USING__PI_only
  msg_Info()<<tm::curon<<std::flush;
#endif
  return Mean();
}

void Foam::CheckTime() const 
{
#ifndef USING__PI_only
  if (rpa.gen.CheckTime()) return;
  msg_Error()<<om::bold<<"Foam::Integrate(..): "
	     <<om::reset<<om::red<<"Timeout. Interrupt integration."
	     <<om::reset<<std::endl;
  kill(getpid(),SIGINT);
#else
#endif
}

double Foam::Update(const int mode)
{
  double sum(0.0), alpha(1.0/(m_channels.size()-m_point.size()));
  if (mode==0) m_sum=m_sum2=m_np=0.0;
  m_max=0.0;
  for (size_t i(0);i<m_channels.size();++i) {
    if (!m_channels[i]->Boundary()) {
      if (mode==1) alpha=m_channels[i]->Alpha();
      if (alpha==0.0) THROW(fatal_error,"Integration domain not covered.");
      sum+=alpha;
      if (mode==0 && m_channels[i]->Points()<m_nopt/3)
	msg_Error()<<METHOD<<"(): "
		   <<"Few points in cell. Increase NOpt."<<std::endl;
      m_np+=m_channels[i]->Points();
      m_sum+=m_channels[i]->Sum()/alpha;
      m_sum2+=m_channels[i]->Sum2()/sqr(alpha);
      m_max=FOAM::Max(m_max,m_channels[i]->Max()/alpha);
    }
  }
  m_smax.pop_back();
  m_smax.push_front(m_max);
  for (size_t i(0);i<m_smax.size();++i) 
    m_max=FOAM::Max(m_max,m_smax[i]);
  if (!IsEqual(sum,1.0)) 
    THROW(fatal_error,"Summation does not agree.");
  double error(dabs(Sigma()/Mean()));
#ifndef USING__PI_only
  msg_Info()<<"  "<<om::bold<<m_vname<<om::reset<<" = "<<om::blue
	    <<Mean()*m_scale<<" "<<m_uname<<om::reset<<" +- ( "
	    <<error*Mean()*m_scale<<" "<<m_uname<<" = "<<om::red
	    <<error*100.0<<" %"<<om::reset<<" )\n  eff = "
	    <<Mean()/m_max*100.0<<" %, n = "<<m_np
	    <<" ( "<<m_np/m_nrealp*100.0<<" % ), "
	    <<m_channels.size()-m_point.size()
	    <<" cells      "<<mm(1,mm::up)<<bm::cr<<std::flush;
#else
  msg_Info()<<"  "<<m_vname<<" = "<<Mean()*m_scale<<" "<<m_uname
	    <<" +- ( "<<error*Mean()*m_scale<<" "<<m_uname<<" = "
	    <<error*100.0<<" % )\n  eff = "
	    <<Mean()/m_max*100.0<<" %, n = "<<m_np
	    <<" ( "<<m_np/m_nrealp*100.0<<" % ), "
	    <<m_channels.size()-m_point.size()
	    <<" cells      "<<mm_up(1)<<bm_cr<<std::flush;
#endif
  return error;
}

double Foam::Point()
{
  Foam_Channel *selected=NULL;
  double disc(ran.Get());
  if (m_asum.empty()) {
    double sum(0.0);
    for (size_t i(0);i<m_channels.size();++i) {
      sum+=m_channels[i]->Alpha();
      if (sum>=disc) {
	selected=m_channels[m_last=i];
	break;
      }
    }
  }
  else {
    size_t l(0), r(m_asum.size()-1), i((l+r)/2);
    double a(m_asum[i]);
    while (r-l>1) {
      if (disc<a) r=i;
      else l=i;
      i=(l+r)/2;
      a=m_asum[i];
    }
    while (m_channels[l]->Boundary()) --l;
    selected=m_channels[m_last=l];
  }
  if (selected==NULL) THROW(fatal_error,"No channel selected.");
  selected->Point(p_function,m_point);
  ++m_nrealp;
  return selected->Weight()/selected->Alpha();
}

double Foam::Point(std::vector<double> &x)
{
  if (x.size()!=m_point.size()) 
    THROW(fatal_error,"Inconsistent dimensions.");
  double weight(Point());
  x=m_point;
  return weight;
}

double Foam::Weight(const std::vector<double> &x) const
{
  if (m_last<m_channels.size() && x==m_point) return 
    m_channels[m_last]->Weight()/m_channels[m_last]->Alpha();
  for (size_t i(0);i<m_channels.size();++i) {
    if (m_channels[i]->Find(x)) {
      return m_channels[i]->Weight()/m_channels[i]->Alpha();
    }
  }
  THROW(critical_error,"Point out of range.");
  return 0.0;
}

void Foam::Split()
{
  if (m_split==0) return;
  msg_Debugging()<<METHOD<<"(): {\n";
  {
    msg_Indent();
    for (size_t i(0);i<m_channels.size();++i) 
      msg_Debugging()<<*m_channels[i];
  }
  msg_Debugging()<<"}"<<std::endl;
  double max(-std::numeric_limits<double>::max()), cur(0.0);
  Foam_Channel *selected(NULL);
  for (size_t i(0);i<m_channels.size();++i) {
    if (!m_channels[i]->Boundary()) {
      if (m_channels[i]->Position()!=std::string::npos) {
	m_channels[i]->SetPosition(std::string::npos);
	m_channels[i]->SelectSplitDimension(m_nosplit);
      }
      m_channels[i]->SetAlpha(0.0);
      cur=m_channels[i]->Loss();
      if (cur>max) {
	max=cur;
	selected=m_channels[i];
      }
    }
  }
  if (selected==NULL) {
    if (m_channels.size()==m_point.size()+1) selected=m_channels.front();
    else THROW(fatal_error,"Internal error.");
  }
  selected->SetAlpha(selected->OldAlpha());
  Foam_Channel 
    *next(new Foam_Channel(this,selected,selected->SplitDimension()));
  m_channels.push_back(next);
  next->SaveAlpha();
  next->SetAlpha(0.5);
  next->SetPosition(1);
  selected->SaveAlpha();
  selected->SetAlpha(0.5);
  selected->SetPosition(0);
  m_apweight/=m_apweight+1.0;
  m_ndiced=(size_t)(selected->Points()+next->Points());
}

bool Foam::Shuffle()
{
  if (m_shuffle==0) {
    for (size_t i(0);i<m_channels.size();++i) 
      if (!m_channels[i]->Boundary()) {
	m_channels[i]->Store();
	m_channels[i]->Reset();
      }
    return true;
  }
  size_t diced(0);
  double norm(0.0), oldnorm(0.0);
  for (size_t i(0);i<m_channels.size();++i) {
    if (!m_channels[i]->Boundary()) {
      double alpha(m_channels[i]->Alpha());
      m_channels[i]->Store();
      if (m_channels[i]->Sum2()!=0.0) {
 	oldnorm+=alpha;
	alpha=sqrt(alpha*m_channels[i]->SSum2()/m_channels[i]->SSum());
	if (!(alpha>0.0)) 
	  THROW(fatal_error,"Invalid weight.");
	m_channels[i]->SetAlpha(alpha);
	norm+=alpha;
 	++diced;
      }
    }
  }
  norm/=oldnorm;
  oldnorm=0.0;
  if (diced==0) THROW(fatal_error,"No channel diced.");
  for (size_t i(0);i<m_channels.size();++i) {
    if (!m_channels[i]->Boundary()) {
      if (m_channels[i]->Sum2()!=0.0)
	m_channels[i]->SetAlpha(m_channels[i]->Alpha()/norm);
      m_channels[i]->Reset();
    }
    m_asum[i+1]=oldnorm+=m_channels[i]->Alpha();
  }
  if (!IsEqual(oldnorm,1.0)) 
    THROW(fatal_error,"Summation does not agree.");
  return true;
}

bool Foam::WriteOut(const std::string &filename) const
{
  std::fstream *file = new std::fstream(filename.c_str(),std::ios::out);
  if (file->bad()) {
    delete file;
    return false;
  }
  bool result=true;
  file->precision(14);
  (*file)<<m_nopt<<" "<<m_nmax<<" "<<m_ncells<<"\n";
  (*file)<<m_error<<" "<<m_scale<<"\n";
  (*file)<<m_sum<<" "<<m_sum2<<" "<<m_max<<" "<<m_np<<"\n";
  (*file)<<m_rmin.size()<<" ";
  for (size_t i(0);i<m_rmin.size();++i) (*file)<<m_rmin[i]<<" ";
  (*file)<<"\n"<<m_rmax.size()<<" ";
  for (size_t i(0);i<m_rmax.size();++i) (*file)<<m_rmax[i]<<" ";
  (*file)<<"\n"<<m_channels.size()<<" {\n";
  std::map<Foam_Channel*,size_t> pmap;
  for (size_t i(0);i<m_channels.size();++i) pmap[m_channels[i]]=i+1;
  for (size_t i(0);i<m_channels.size();++i) 
    if (!m_channels[i]->WriteOut(file,pmap)) result=false;
  (*file)<<"}\n"<<std::endl;
  delete file;
  return result;
}

bool Foam::ReadIn(const std::string &filename)
{
#ifndef USING__PI_only
  msg_Debugging()<<METHOD<<"(\""<<filename<<"\"):"<<std::endl;
#endif
  if (!m_channels.empty()) return false;
  std::fstream *file = new std::fstream(filename.c_str(),std::ios::in);
  if (!file->good()) {
    msg_Info()<<METHOD<<"(\""<<filename<<"\"): "
	      <<"Cannot find file."<<std::endl;
    delete file;
    return false;
  }
  std::string dummy;
  file->precision(14);
  (*file)>>m_nopt>>m_nmax>>m_ncells;
  (*file)>>m_error>>m_scale;
  (*file)>>m_sum>>m_sum2>>m_max>>m_np;
  if (file->eof()) {
    delete file;
    return false;
  }
  size_t size;
  (*file)>>size;
  m_rmin.resize(size);
  for (size_t i(0);i<m_rmin.size();++i) (*file)>>m_rmin[i];
  (*file)>>size;
  m_rmax.resize(size);
  for (size_t i(0);i<m_rmax.size();++i) (*file)>>m_rmax[i];
  (*file)>>size>>dummy;
  if (file->eof() || dummy!="{") {
    msg_Error()<<METHOD<<"("<<filename<<"): Data error.";
    delete file;
    return false;
  }
  Foam_Channel::CreateRoot(this,m_rmin,m_rmax,m_channels);
  std::map<size_t,Foam_Channel*> pmap;
  pmap[0]=NULL;
  for (size_t i(0);i<size;++i) {
    if (i<m_rmin.size()) {
      pmap[i+1]=m_channels[i];
    }
    else {
      if (i<size-1) m_channels.
	push_back(new Foam_Channel(this,m_channels[0],0));
      pmap[i+1]=m_channels[i];
    }
  }
  bool result=true;
  for (size_t i(0);i<m_channels.size();++i) {
    if (!m_channels[i]->ReadIn(file,pmap)) result=false;
    msg_Tracking()<<*m_channels[i]<<std::endl;
  }
  for (size_t i(0);i<m_channels.size();++i) m_channels[i]->SetWeight();
  (*file)>>dummy;
  if (file->eof() || dummy!="}") {
    msg_Error()<<METHOD<<"("<<filename<<"): Data error.";
    delete file;
    return false;
  }
  delete file;
  m_point.resize(m_rmin.size());
  return result;
}

void Foam::Split(const size_t dim,
		 const std::vector<double> &pos,const bool nosplit)
{
  if (m_channels.empty()) 
    THROW(critical_error,"No cells. Call Initialize() first.");
  m_nosplit[dim]=nosplit;
  const std::vector<Foam_Channel*> channels(m_channels);
  for (size_t i(pos.size());i>0;--i) {
    for (size_t j(0);j<channels.size();++j) {
      if (channels[j]->Boundary()) continue;
      Foam_Channel *next = 
	new Foam_Channel(this,channels[j],dim,pos[i-1]);
      m_channels.push_back(next);    
    }
  }
  const double alpha(1.0/(m_channels.size()-m_point.size()));
  for (size_t i(0);i<m_channels.size();++i) {
    m_channels[i]->SetAlpha(alpha);
    m_channels[i]->SaveAlpha();
  }
}

std::pair<double,double> Foam::Loss(const Foam_Channel *c,int pos) const
{
  const Foam_Channel::Point_Vector &points(c->GetPoints());
  if (points.empty()) THROW(fatal_error,"No data points");
  if (pos<0) pos=points.size();
  double s2l(0.0), s2r(0.0), w(c->Weight());
  if (m_mode==1) {
    for (size_t i(0);i<pos;++i) s2l=FOAM::Max(points[i].second*w,s2l);
    for (size_t i(pos);i<points.size();++i) s2r=FOAM::Max(points[i].second*w,s2r);
  }
  else {
    for (size_t i(0);i<pos;++i) s2l+=sqr(points[i].second*w);
    for (size_t i(pos);i<points.size();++i) s2r+=sqr(points[i].second*w);
  }
  return std::make_pair(s2l,s2r);
}

void Foam::SetMin(const std::vector<double> &min) 
{ 
  m_rmin=min; 
  for (size_t i(0);i<min.size();++i) m_nosplit[i]=false;
}

void Foam::SetMax(const std::vector<double> &max) 
{
  m_rmax=max; 
  for (size_t i(0);i<max.size();++i) m_nosplit[i]=false;
}

