/*
 * Class that runs Pythia and analyses events
 *
 */

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include "Pythia8/Pythia.h"


/* A class for random doubles */
class RandomNumber {
public:
    
    RandomNumber() : m_rd(), m_gen(m_rd()), m_uniform(0, 1), m_normal(0, 1) {}
    
    /* Get random uniform number between 0 and 1 */
    double get_uniform() {return m_uniform(m_gen);}

    /* Get random normal number with location = 0, scale = 1 */
    double get_normal() {return m_normal(m_gen);}
    
private:
    std::random_device m_rd;
    std::mt19937 m_gen;
    std::uniform_real_distribution<> m_uniform;
    std::normal_distribution<> m_normal;
};




/* The analysis */
class Analysis {

public:

    /* Enum for choice of rest frame
     * TRUTH:   True ditau rest frame
     * VISIBLE: Rest frame of visible ditau decay products (no neutrinos)
     * CHARGED: Rest frame of charged ditau products
    */
    enum RestFrame {LAB, TRUTH, VISIBLE, CHARGED};


    /* Constructor
     * @param cmndfile: Name of process command file
     * @param random_seed: See http://home.thep.lu.se/Pythia/pythia82html/RandomNumberSeed.html
     */
    Analysis(std::string cmndfile, int target, std::string restframe,
             int random_seed = -1, bool apply_smearing = false, bool debug = false);

    /* Destructor */
    ~Analysis();
    
    /* Set tau cuts */
    void set_cuts(float min_leading_pt,
                  float min_subleading_pt,
                  float max_eta,
                  float deltaR,
                  float min_MET,
                  float mass_lower,
                  float mass_upper) {
        m_min_leading_pt = min_leading_pt;
        m_min_subleading_pt = min_subleading_pt;
        m_max_eta = max_eta;
        m_min_deltaR = deltaR;
        m_min_MET = min_MET;
        m_mass_lower = mass_lower;
        m_mass_upper = mass_upper;
        std::cout.precision(1);
        std::cout << "Event selection cuts:" << std::endl;
        std::cout << "Leading tau pT > " << m_min_leading_pt << std::endl;
        std::cout << "Subleading tau pT > " << m_min_subleading_pt << std::endl;
        std::cout << "Pseudorapidity < " << m_max_eta << std::endl;
        std::cout << "Delta R < " << m_min_deltaR << std::endl;
        std::cout << "Missing E_T < " << m_min_MET << std::endl;
        std::cout << "Mass range [" << m_mass_lower << ", " << m_mass_upper << "]" << std::endl;
        std::cout.precision(0);
        
    }
    
    /* Get current number of accepted events */
    size_t get_n_accepted_events() {
        return n_accepted_events;
    }

    /* Get number of events which passed mass selection */
    size_t get_n_passed_mass_cuts() {
        return m_n_passed_mass_cuts;
    }
    
    /* Get total number of tried events */ 
    size_t get_n_tried_events() {
        return n_total_events;
    }
    
    /* Get the event selection efficiency */
    double get_efficiency() {
        if (n_total_events > 0)
            m_efficiency = static_cast<float>(n_accepted_events) / static_cast<float>(n_total_events);
        else
            m_efficiency = 0;
        return m_efficiency;
    }
    
    
    /* Get final cross section */ 
    double get_cross_section() {
        return m_pythia.info.sigmaGen();
    }
    
    
    /* Require a higgs -> tautau decay  to be present in the event 
     * Speeds up the analysis, but must be turned off if generating backgrounds
     */
    void require_htautau(bool yes = true) {
        m_require_htautau = yes;
        std::cout << "Skipping events with no H->tautau decay" << std::endl;
    }

    /* Return vector with event features */
    std::vector<double> process_event();

    /* Scan past bremsstrahlung, AND past duplicate entries
     * ip: index if input tau
     * returns: index of decaying tau
     */
    int scan_past_brem(int ip);

    /* Return unit vector */
    Pythia8::Vec4 unit(const Pythia8::Vec4 vec);

    /* Compute charged to neutral energy ratio */
    double upsilon(const Pythia8::Vec4 charged, const Pythia8::Vec4 neutral);

    /* Tau decay acoplanarity angle (adjusted for T-odd triple correlation, and
     * sign of upsilon product)
     */
    double acoplanarity(const Pythia8::Vec4 pim, const Pythia8::Vec4 pi0m,
                        const Pythia8::Vec4 pip, const Pythia8::Vec4 pi0p,
                        bool return_triple_corr = false);

    /* Compute missing transverse energy in the event by summing up neutrino
     * energies
     * returns: Full MET 4-vector in lab frame
     */
    Pythia8::Vec4 sum_met();

    /* Compute the total transverse mass */
    double total_transverse_mass(const Pythia8::Vec4 tau1,
                                 const Pythia8::Vec4 tau2,
                                 const Pythia8::Vec4 met);
    
    /* Compute the track impact parameter vector */
    Pythia8::Vec4 impact_param_vector(const Pythia8::Particle pion,
                                      const Pythia8::Vec4 privertex);
    
    /* Check for a b-quark from top decay */
    bool b_tag(const int tauindex);
    
    /* Check for electrons, muons in the event */
    bool lepton_veto(float minpt);

    /* Smear fourvector (in place) */
    Pythia8::Vec4 smear_fourvector(Pythia8::Vec4 pin, float angular_res, float energy_res);



private:

    Pythia8::Pythia m_pythia;
    RestFrame m_restframe;
    
    /* Random number generator */
    RandomNumber m_rand;
    
    /* Target value */
    int m_target;
    
    /* Tau cuts */
    float m_min_leading_pt;
    float m_min_subleading_pt;
    float m_max_eta;
    float m_min_deltaR;
    float m_min_MET;
    float m_mass_lower;
    float m_mass_upper;
    
    /* Cut flow numbers */
    size_t m_n_pt;
    size_t m_n_samesign;
    size_t m_n_eta;
    size_t m_n_deltaR;
    size_t m_n_MET;
    size_t m_n_mass;
    size_t m_n_btag;
    size_t m_n_leptonveto;
    size_t m_n_passed_mass_cuts;

    /* Switch to enable some debug output */
    bool m_debug;
    
    /* Skip event if no higgs->tautau found */
    bool m_require_htautau;
    
    /* For event filtering efficiency */
    size_t n_accepted_events;
    size_t n_total_events;
    double m_efficiency;

    /* Counters to log number of failed events */
    unsigned int n_pythia_failures;
    unsigned int n_missing_taus;
    unsigned int n_tau_decay_errors;

    /* Switch for detector reconstruction smearing */
    bool m_apply_smearing;
};
