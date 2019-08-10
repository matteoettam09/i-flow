%module Foam
%{
#define SWIG_FILE_WITH_INIT
#include "Foam.H"
#include "Random.H"
#include "MathTools.H"
#include "Tools.H"
#include <fstream>
#include <stddef.h>

using namespace ATOOLS;
%}

namespace ATOOLS {

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
    void Point();
    void   Point(std::vector<double> &x);
    double Weight(const std::vector<double> &x) const;
    bool WriteOut(const std::string &filename) const;
    bool ReadIn(const std::string &filename);
    void Reserve(const std::string &key,
		 const size_t n,const size_t nprev=0);
    void Split(const std::string &key,const size_t nprev,
	       const std::vector<double> &pos,const bool nosplit=true);
    const double *const Reserved(const std::string &key) const;
    inline void SetNOpt(const long unsigned int &nopt) { m_nopt=nopt; }
    inline void SetNMax(const long unsigned int &nmax) { m_nmax=nmax; }
    inline void SetNCells(const size_t &ncells) { m_ncells=ncells; }
    inline void SetError(const double &error) { m_error=error; }
    inline void SetScale(const double &scale) { m_scale=scale; }
    inline void SetMode(const imc::code mode) { m_mode=mode; }
    inline void SetVariableName(const std::string &vname) { m_vname=vname; }
    inline void SetUnitName(const std::string &uname) { m_uname=uname; } 
    inline void SetSplitMode(const size_t split) { m_split=split; }
    inline void SetShuffleMode(const size_t shuffle) { m_shuffle=shuffle; }
    inline void SetFunction(Foam_Integrand *const function)
    { p_function=function; }
    inline Foam_Integrand *const Function() const
    { return p_function; }
    inline imc::code Mode() const { return m_mode; }
    inline rmc::code RunMode() const { return m_rmode; }
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
