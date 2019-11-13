%module Foam
%{
#define SWIG_FILE_WITH_INIT
#include "Foam.H"
#include "Random.H"
#include "MathTools.H"
#include "Tools.H"
#include <fstream>
#include <stddef.h>

using namespace FOAM;
%}

%typemap(typecheck) std::vector<double> { $1 = PySequence_Check($input); }
%typemap(in) std::vector<double> {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence");
    return NULL;
  }
  else {
    $1 = std::vector<double>(PySequence_Length($input));
    for (int i = 0; i < $1.size(); i++) {
      PyObject *o = PySequence_GetItem($input,i);
      if (PyNumber_Check(o)) $1[i] = PyFloat_AsDouble(o);
      else {
	PyErr_SetString(PyExc_ValueError,"Sequence elements must be numbers");
	return NULL;
      }
    }
  }
}

namespace FOAM {

  class Foam_Integrand {
  public:

    virtual ~Foam_Integrand();
    virtual double operator()(const std::vector<double> &point);
    virtual void AddPoint(const double value,const double weight,
			  const int mode=0);
    virtual void FinishConstruction(const double apweight);

  };

  class Foam {
  public:

    Foam();
    ~Foam();
    void SetDimension(const size_t dim);
    void SetMin(const std::vector<double> &min);
    void SetMax(const std::vector<double> &max);
    void   Reset();
    void   Initialize();
    double Integrate(PyObject *const function);
    double Point();
    double Point(std::vector<double> &x);
    double Weight(const std::vector<double> &x) const;
    %extend{
    bool WriteOut(const char *filename) const
    { return self->WriteOut(std::string(filename)); }
    bool ReadIn(const const char *filename)
    { return self->ReadIn(std::string(filename)); }
    }
    void Split(const size_t n,std::vector<double> pos,const bool nosplit=true);
    inline void SetNOpt(const long unsigned int &nopt) { m_nopt=nopt; }
    inline void SetNMax(const long unsigned int &nmax) { m_nmax=nmax; }
    inline void SetNCells(const size_t &ncells) { m_ncells=ncells; }
    inline void SetError(const double &error) { m_error=error; }
    inline void SetScale(const double &scale) { m_scale=scale; }
    inline void SetStorePoints(const int store) { m_store=store; }
    %extend{
    inline void SetVariableName(const char *vname)
    { self->SetVariableName(std::string(vname)); }
    inline void SetUnitName(const char *uname)
    { self->SetUnitName(std::string(uname)); } 
    }
    inline void SetSplitMode(const size_t split) { m_split=split; }
    inline void SetShuffleMode(const size_t shuffle) { m_shuffle=shuffle; }
    inline void SetCutFactor(const double cutfac) { m_cutfac=cutfac; }
    inline void SetFunction(Foam_Integrand *const function)
    { p_function=function; }
    inline Foam_Integrand *const Function() const
    { return p_function; }
    inline int StorePoints() const { return m_store; }
    inline double Mean() const { return m_sum/m_np; }
    inline double Max() const { return m_max; }
    inline double Points() const { return m_np; }
    inline double Variance() const    
    { return (m_sum2-m_sum*m_sum/m_np)/(m_np-1.0); }
    inline double Sigma() const  
    { return sqrt(Variance()/m_np); }
    inline double APrioriWeight() const { return m_apweight; }
  };

}
