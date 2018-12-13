#include "THDM.h"
#include "Constraints.h"
#include "DecayTable.h"
#include "HBHS.h"

#include "TF1.h"

#include <iostream>
#include <math.h>

using namespace std;


/* Struct to store relevant model values */ 
struct Model {
    int allowed;
    float lambda1;
    float lambda2;
    float br_H_tautau;
    float br_A_tautau;
    char outfilename[100];
};



/* Run 2HDMC for a given set of parameters */
Model calcphys(float mH, float mA, float m_12_2, float tanb, bool write_lha=false) {
   
    // Input
    float mh = 125;
    float mHp = mA;
    float sba = 1.0;
    float l6 = 0;
    float l7 = 0;
 
    // Output
    Model thismodel;
    thismodel.allowed = false;

    // 2HDMC classes
    THDM thdm;
    SM sm;
    thdm.set_SM(sm);

    bool pset = thdm.set_param_phys(mh, mH, mA, mHp, sba, l6, l7, m_12_2, tanb);

    if (!pset) return thismodel;

    thdm.set_yukawas_type(4);

    // Get model parameters
    double lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, tb, m12;
    thdm.get_param_gen(lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, tb, m12);
    thismodel.lambda1 = lambda1;
    thismodel.lambda2 = lambda2;

    // Check constraints
    Constraints check(thdm);
    thismodel.allowed = static_cast<int>(check.check_unitarity()) 
                        + static_cast<int>(check.check_perturbativity()) 
                        + static_cast<int>(check.check_stability());
    if (thismodel.allowed < 3) return thismodel;
    
    // Check HiggsBounds
    #if defined HiggsBounds
    HB_init();
    HB_set_input_effC(thdm);
    int hbres[6];
    double hbobs[6];
    int hbchan[6];
    int hbcomb[6];
    HB_run_full(hbres, hbchan, hbobs, hbcomb);
    if (hbobs[0] > 1) return thismodel;
    #endif
    
    
    // Get decay widths to tau tau
    DecayTable table(thdm);
    thismodel.br_H_tautau = table.get_gamma_hll(2, 3, 3)/table.get_gammatot_h(2);
    thismodel.br_A_tautau = table.get_gamma_hll(3, 3, 3)/table.get_gammatot_h(3);
    
    // Write output as LHA
    if (write_lha) {
        char outname[100];
        sprintf(outname, "massH_%d_massA_%d_tanb_%f_m12_%d.lha", static_cast<int>(mH), static_cast<int>(mA), tanb, static_cast<int>(m_12_2));
        thdm.write_LesHouches(outname, true, true, true, true);
        strcpy(thismodel.outfilename, outname);
    }
    
    return thismodel;
}



/* Express model validity as a function which can be optimised
 * This is a piecewise constant function, so to ensure we always have a gradient,
 * add the absolute value of lambda1 (which has to be small for a model which
 * conserves unitarity). Lambda2 is also somewhat important, so add a dash of
 * that too
 */
float fn_allowed_model(double *m_12, double *pars) {
    
    // pars[0]: mH
    // pars[1]: mA
    // pars[2]: tanb
    
    Model res = calcphys(pars[0], pars[1], m_12[0], pars[2]);
    
    return (-10*res.allowed) + std::fabs(std::abs(std::abs(res.lambda1)-1) + 0.5*std::abs(res.lambda2));

}


/* Return m_12^2 value that gives a valid point */
float minimize_neg_validity(float mH, float mA, float tanb) {
    
    float range_low = -10e3;
    float range_high = 10e5;
    
    TF1 fn = TF1("fn", fn_allowed_model, 100, range_low, range_high);
    fn.SetParameters(mH, mA, tanb);
    
    // Find m12_^2 corresponing to a valid point with minimum abs(lambda1), using Brent's method
    float min_m12 = fn.GetMinimumX(range_low, range_high, 1.0e-6);
    
    return min_m12;
}


/* Starting from a point know to be valid, search for an entire interval of valid
 * points 
 * 
 * search_direction: if -1, get lower bound. If +1, get upper bound
 */
float find_allowed_interval(float mH, float mA, float tanb, float m12_init, float search_direction) {
    
    // Scan left
    float m12 = m12_init;
    std::vector<float> jump = {10000, 1000, 100, 10, 1};
    
    Model test_point = calcphys(mH, mA, m12, tanb);
    int step = 0;
    float current_m12;
    
    while (true) {
        
        current_m12 = m12 + search_direction*jump[step];
        test_point = calcphys(mH, mA, current_m12, tanb);
        
        if (test_point.allowed != 3) {
            step++;
            if (step >= jump.size()) {
                break;
            }
        }
        else {
            m12 = current_m12;
        }
    }
    
    //std::cout << "Limit: " << m12 << std::endl;
    
    return m12;
}



/* MAIN FUNCTION TO BE CALLED FROM RATIO_SCANNER.PY */
int main(int argc, char* argv[]) {
    
    float  mH_in = atof(argv[1]);   // H mass
    double mA_in = atof(argv[2]);   // A mass
    double tb_in = atof(argv[3]);   // tan beta
    
    
    // Find a low m_12^2, which is likely to give unitarity
    float m_12_2 = minimize_neg_validity(mH_in, mA_in, tb_in);
    
    // Write LHA file for this point 
    Model model = calcphys(mH_in, mA_in, m_12_2, tb_in, true);
    std::cout << model.outfilename << std::endl;
    
    // Check validity
    if (model.allowed == 3) return 0;   // success
    
    return -1;  // failure

    
}

