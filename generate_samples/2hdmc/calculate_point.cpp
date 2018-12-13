#include "THDM.h"
#include "SM.h"
#include "HBHS.h"
#include "Constraints.h"
#include "DecayTable.h"
#include <iostream>


using namespace std;


// Input parameters must be specified in the correct order listed below
int main(int argc, char* argv[]) {    
    
    if (argc < 12) {
        cout << "Too few arguments ("<< argc << ")" << endl;
        return -1;
    }
    
    // Set parameters of the 2HDM in the 'physical' basis
    double mh       = atof(argv[1]);
    double mH       = atof(argv[2]);
    double mA       = atof(argv[3]);
    double mC       = atof(argv[4]);
    double sba      = atof(argv[5]);
    double lambda_6 = atof(argv[6]);
    double lambda_7 = atof(argv[7]);
    double m12_2    = atof(argv[8]);
    double tb       = atof(argv[9]);
    char* outname   = argv[10];
    
    int ignore_higgsbounds = atoi(argv[11]);

    // Reference SM Higgs mass for EW precision observables
    double mh_ref = 125.;
    
    // Create SM and set parameters
    SM sm;
    sm.set_qmass_pole(6, 172.5);		
    sm.set_qmass_pole(5, 4.75);		
    sm.set_qmass_pole(4, 1.42);	
    sm.set_lmass_pole(3, 1.77684);	
    sm.set_alpha(1./127.934);
    sm.set_alpha0(1./137.0359997);
    sm.set_alpha_s(0.119);
    sm.set_MZ(91.15349);
    sm.set_MW(80.36951);
    sm.set_gamma_Z(2.49581);
    sm.set_gamma_W(2.08856);
    sm.set_GF(1.16637E-5);
    
    // Create 2HDM and set SM parameters
    THDM model;
    model.set_SM(sm);
    
    bool pset = model.set_param_phys(mh,mH,mA,mC,sba,lambda_6,lambda_7,m12_2,tb);
    
    if (!pset) {
      cerr << "The specified parameters are not valid" << endl;
      return -2;
    }

    // Set Yukawa couplings to type I
    model.set_yukawas_type(1);

    // Prepare to calculate observables
    Constraints constr(model);

    double S,T,U,V,W,X;   

    constr.oblique_param(mh_ref,S,T,U,V,W,X);

    bool cstab = constr.check_stability();
    bool cunit = constr.check_unitarity();
    bool cpert = constr.check_perturbativity();
    
    if (!cstab || !cunit || !cpert) {
        cerr << "Error:" << endl;
        cerr << "Potential stability: " << cstab << endl;
        cerr << "Tree-level unitarity: " << cunit << endl;
        cerr << "Perturbativity: " << cpert << endl;
        return -3;
    }
    
    // HiggsBounds
    HB_init();
    //HS_init();
    HB_set_input_effC(model);
    
    int hbres[6];
    double hbobs[6];
    int hbchan[6];
    int hbcomb[6];  
    
    HB_run_full(hbres, hbchan, hbobs, hbcomb);
    if (hbobs[0] > 1) {
        cerr << "Model excluded by HiggsBounds (obs = " << hbobs[0] << ")" << endl;
        if (!ignore_higgsbounds) return -4;
    }
    
    // Prepare to calculate decay widths
    //DecayTable table(model);
    
    // Write output to LesHouches file
    model.write_LesHouches(outname, 1, 0, 1, 1);
    
    HB_finish();
    
    return 0;
}
