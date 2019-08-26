#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Math/Random.H"
#include "SHERPA/Initialization/Initialization_Handler.H"
#include "SHERPA/PerturbativePhysics/Matrix_Element_Handler.H"
#include "PHASIC++/Process/Process_Base.H"
#include "PHASIC++/Main/Process_Integrator.H"
#include "PHASIC++/Main/Phase_Space_Handler.H"
#include "ATOOLS/Org/Run_Parameter.H"

#include "Foam.H"

#include "Tools.H"
#include <termios.h>
#include <unistd.h>

using namespace FOAM;
using namespace SHERPA;
using namespace ATOOLS;

class MyRNG : public ATOOLS::External_RNG {
    public:
        MyRNG() : counter(0) {};
        void Init(int npart) {
            rands.resize(3*npart - 4 + npart + 1, 0.5);
        }
        double Get() {
            if(rands.empty()) return ran -> Get(1);
            if(counter >= rands.size()) abort();
            return rands[counter++];
        }
        std::vector<double> rands;
        int counter;
};

static MyRNG* rng;

// this makes MyRNG loadable in Sherpa
DECLARE_GETTER(MyRNG,"MyRNG",External_RNG,RNG_Key);
External_RNG *ATOOLS::Getter<External_RNG,RNG_Key,MyRNG>::operator()(const RNG_Key &) const
{ rng = new MyRNG(); return rng; }
// this eventually prints a help message
void ATOOLS::Getter<External_RNG,RNG_Key,MyRNG>::PrintInfo(std::ostream &str,const size_t) const
{ str<<"my RNG interface"; }

class CallFoam : public Foam_Integrand {
    public:
        CallFoam() {
            sherpa = new Sherpa();
            char** argv = new char*[1];
            argv[0] = new char[1];
            argv[0][0] = ' ';
            sherpa -> InitializeTheRun(1, argv);
            std::cout << "NPart: "<< sherpa -> GetInitHandler() -> GetMatrixElementHandler() -> AllProcesses().front() -> NOut() << std::endl;
            PHASIC::Process_Base* proc = sherpa -> GetInitHandler() -> GetMatrixElementHandler() -> AllProcesses().front();
            PHASIC::Process_Integrator* p_int = proc -> Integrator();
            rng -> Init(proc -> NOut());
            p_int->Reset();
            SP(PHASIC::Phase_Space_Handler) psh(p_int->PSHandler());
            psh->InitCuts();
            if (p_int->ISR())
              p_int->ISR()->SetSprimeMin(psh->Cuts()->Smin());
            psh->CreateIntegrators();
            psh->InitIncoming();
            delete argv[0];
            delete argv;
        }

        double operator()(const std::vector<double>& point) {
            rng -> counter = 0;
            for(size_t irand = 0; irand < point.size(); ++irand) {
                rng -> rands[irand] = point[irand];
            }
                return sherpa -> GetInitHandler() -> GetMatrixElementHandler() -> AllProcesses().front() -> Integrator() -> PSHandler() -> Differential() * rpa -> Picobarn();
        }

        ~CallFoam() {
            delete sherpa;
        };

        int npart() {
            return sherpa -> GetInitHandler() -> GetMatrixElementHandler() -> AllProcesses().front() -> NOut();
        }


    private:
        Sherpa *sherpa;
};

int main(int argc,char **argv)
{
  CallFoam camel;

  Foam integrator;
  // set dimension
  integrator.SetDimension(3*camel.npart()-4);
  // set max cell number
  integrator.SetNCells(500);
  // set point number between optimization steps
  integrator.SetNOpt(1000);
  // set max point number
  integrator.SetNMax(2000000);
  // set error
  integrator.SetError(5.0e-4);
  // set variance optimization mode
  integrator.SetMode(0);
  // integrate
  integrator.Initialize();
  PRINT_INFO("Integrate camel");
  integrator.Integrate(&camel);
  return 0;
}

