#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Math/Random.H"
#include "SHERPA/Initialization/Initialization_Handler.H"
#include "SHERPA/PerturbativePhysics/Matrix_Element_Handler.H"
#include "PHASIC++/Process/Process_Base.H"
#include "PHASIC++/Main/Process_Integrator.H"
#include "PHASIC++/Main/Phase_Space_Handler.H"
#include "ATOOLS/Org/Run_Parameter.H"

using namespace tensorflow;
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

REGISTER_OP("CallSherpa")
    .Input("random: float64")
    .Output("xsec: float64");

class CallSherpaOp : public OpKernel {
    public:
        explicit CallSherpaOp(OpKernelConstruction* context) : OpKernel(context) {
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

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor and its shape
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<double>();
            TensorShape input_shape = input_tensor.shape();
            size_t nBatch = input_shape.dim_size(0);
            size_t nRandom = input_shape.dim_size(1);

            // Create an output tensor
            Tensor* output_tensor = NULL;
            TensorShape output_shape = input_shape;
            output_shape.RemoveDim(1);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto output_flat = output_tensor->flat<double>();

            // Calculate the cross-sections
            for(size_t ibatch = 0; ibatch < nBatch; ++ibatch) {
                rng -> counter = 0;
                for(size_t irand = 0; irand < nRandom; ++irand) {
                    rng -> rands[irand] = input(ibatch*nRandom + irand);
                }
                output_flat(ibatch) = sherpa -> GetInitHandler() -> GetMatrixElementHandler() -> AllProcesses().front() -> Integrator() -> PSHandler() -> Differential() * rpa -> Picobarn();
            }
        }

        ~CallSherpaOp() {
            delete sherpa;
        };

    private:
        Sherpa *sherpa;
};

REGISTER_KERNEL_BUILDER(Name("CallSherpa").Device(DEVICE_CPU), CallSherpaOp);
