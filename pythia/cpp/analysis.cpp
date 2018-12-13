#include "analysis.h"
#include <math.h>
#include <map>

using namespace Pythia8;

// -----------------------------------------------------------------------------
Analysis::Analysis(std::string cmndfile, int target, std::string restframe, int random_seed, bool apply_smearing, bool debug)
: m_pythia(), m_rand() {
    
    // Print the input parameters we received
    std::cout << "In constructor: Got parameters" << std::endl;
    std::cout << "\tCommand file: " << cmndfile << std::endl;
    std::cout << "\tTarget: " << target << std::endl;
    std::cout << "\tRest frame: " << restframe << std::endl;
    std::cout << "\tRandom seed: " << random_seed << std::endl;
    std::cout << "\tApply smearing: " << apply_smearing << std::endl;

    m_target = target;

    // Restframe selection
    if (restframe == "lab") m_restframe = LAB;
    else if (restframe == "truth") m_restframe = TRUTH;
    else if (restframe == "visible") m_restframe = VISIBLE;
    else if (restframe == "charged") m_restframe = CHARGED;
    else {
        std::cout << "Invalid rest frame: " << restframe << std::endl;
        exit(-1);
    }

    // Detector smearing
    if (apply_smearing) m_apply_smearing = true;
    
    // Enable debugging 
    if (debug){
        m_debug = true;
        std::cout << "Enabling debug output" << std::endl;
    }

    // Set up Pythia
    m_pythia.readFile(cmndfile);
    if (random_seed >= 0) {
        m_pythia.readString("Random:setSeed = on");
        m_pythia.readString("Random:seed = " + std::to_string(random_seed));
    }
    if (!m_debug) {
        m_pythia.readString("Init:showChangedSettings = off");
        m_pythia.readString("Next:numberShowEvent = 0");
        m_pythia.readString("Init:showChangedParticleData = off");
        m_pythia.readString("Next:numberShowProcess = 0");
    }
    m_pythia.init();
    
    // Initialize counters, switches
    n_pythia_failures = 0, n_missing_taus = 0, n_tau_decay_errors = 0;
    n_accepted_events = 0; n_total_events = 0;
    m_n_passed_mass_cuts = 0;
    m_efficiency = 0;
    m_debug = false;
    m_require_htautau = false;
    
    m_n_pt = 0; m_n_eta = 0; m_n_deltaR = 0; m_n_MET = 0; m_n_mass = 0;
    m_n_btag = 0; m_n_leptonveto = 0; m_n_samesign = 0;


}

// -----------------------------------------------------------------------------
Analysis::~Analysis() {

    // Sayonara
    m_pythia.stat();

    std::cout << std::endl;
    std::cout << "Number of critical Pythia failures:\t\t" << n_pythia_failures
              << std::endl;
    std::cout << "Number of events with missing taus:\t\t" << n_missing_taus
              << std::endl;
    std::cout << "Number of events with unexpected tau decays:\t"
              << n_tau_decay_errors << std::endl << std::endl;
    
    // Compute efficiency, if not already done
    get_efficiency();
    
    std::cout << "Event selection efficiency: " << m_efficiency << std::endl;
    std::cout << "Cross section: " << get_cross_section() << " mb " << std::endl;
    
    // Print cut flow
    std::cout << std::endl << "Cut flow:" << std::endl;
    std::cout << " same-sign:\t" << m_n_samesign << std::endl;
    std::cout << " b-tag:\t\t" << m_n_btag << std::endl;
    std::cout << " lepton veto:\t" << m_n_leptonveto << std::endl;
    std::cout << " mass range:\t" << m_n_mass << std::endl;
    std::cout << " pT:\t\t" << m_n_pt << std::endl;
    std::cout << " max eta:\t" << m_n_eta << std::endl;
    std::cout << " delta R:\t" << m_n_deltaR << std::endl;
    std::cout << " MET:\t\t" << m_n_MET << std::endl << std::endl;
    
    std::cout << " Generated:\t" << n_total_events << std::endl;
    std::cout << " Passed mass selection:\t" << m_n_passed_mass_cuts << std::endl;
    std::cout << " Accepted:\t" << n_accepted_events << std::endl << std::endl;
    
    

}


// -----------------------------------------------------------------------------
std::vector<double> Analysis::process_event() {

    std::vector<double> eventdata;
    
    // Generate ev3nt
    if (!m_pythia.next()) {
        n_pythia_failures++;
        return eventdata;
    }
    
    // Sum up tried events
    ++n_total_events;
    
    // If we require a H->tautau vertex to be present: 
    // Scan for it, skip event on failure
    if (m_require_htautau) {
        bool success = false;
        
        std::set<int> ids = {25, 35, 36};   // higgs pdgIDs
        for (int i = 0; i < m_pythia.event.size(); i++) {
            if (ids.find(m_pythia.event.at(i).id()) != ids.end()) { // check if id in list
                Particle higgs = m_pythia.event.at(scan_past_brem(i));
                Particle daughter1 = m_pythia.event.at(higgs.daughter1());
                Particle daughter2 = m_pythia.event.at(higgs.daughter2());
                if (abs(daughter1.id()) == 15 && abs(daughter2.id()) == 15) {
                    success = true;
                }
                break;
            }
        }
        
        if (m_debug) std::cout << "Found valid H->tautau decay: " 
                               << (success ? "YES" : "NO") << std::endl;
        
        if (!success) return eventdata;
    }
    
    
    // Get all taus in event record
    std::vector<std::pair<int, float> > taus_pt;
    
    for (int i = 0; i < m_pythia.event.size(); i++) {
        if (m_pythia.event.at(i).idAbs() == 15 && m_pythia.event.at(i).daughterList().size() == 3) {
            // It's a hadronically decaying tau
            taus_pt.push_back(std::pair<int, float>(i, m_pythia.event.at(i).pT()));
        }
    }
    
    if (taus_pt.size() < 2) return eventdata;
    
    // Sort on pT
    sort(taus_pt.begin(), taus_pt.end(),
        [](const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
             return lhs.second > rhs.second;
         });
    
    
    if (m_debug) {
        std::cout << "Leading tau pT = " << taus_pt[0].second << std::endl;
        std::cout << "Subleading tau pT = " << taus_pt[1].second << std::endl;
    }
    
    // Reject event if two leading taus have same sign
    if (m_pythia.event.at(taus_pt[0].first).charge() == m_pythia.event.at(taus_pt[1].first).charge()) {
        m_n_samesign++;
        return eventdata;
    }
    
    // Assign plus/minus
    int taup_index = taus_pt[0].first, taum_index = taus_pt[1].first;
    if (m_pythia.event.at(taum_index).id() == -15) std::swap(taum_index, taup_index);


    if (taup_index == 0 || taum_index == 0) {
        n_missing_taus++;
        if (m_debug) {
            std::cout << "Missing taus in event record:";
            std::cout << "tau+ index = " << taup_index << ", tau- index = "
                      << taum_index << std::endl;
        }
        return eventdata;
    }
    

    // Move past brem, and potential duplicates (like 15 -> 15)
    taum_index = scan_past_brem(taum_index);
    taup_index = scan_past_brem(taup_index);

    Particle tauminus = m_pythia.event.at(taum_index);
    Particle tauplus = m_pythia.event.at(taup_index);

    // Debug
    if (tauminus.id() != 15 || tauplus.id() != -15) {
        n_missing_taus++;
        return eventdata;
    }
    if (tauminus.daughterList().size() != 3 || tauplus.daughterList().size() != 3) {
        if (m_debug) {
            std::cout << "Error with tau decays:" << std::endl;
            std::cout << " tau- decayed to ";
            for (auto p : tauminus.daughterList()) {
                std::cout << m_pythia.event.at(p).id() << " ";
            }
            std::cout << std::endl;
            std::cout << " tau+ decayed to ";
            for (auto p : tauplus.daughterList()) {
                std::cout << m_pythia.event.at(p).id() << " ";
            }
            std::cout << std::endl;
        }
        n_tau_decay_errors++;
        return eventdata;
    }
    
    
    // b-tagging
    // For each tau, if it originated from a top with an accompanying b quark,
    // compute the probability of identifying it
    float tag_prob = 0.75;
    if (b_tag(taum_index)) {
        if (m_rand.get_uniform() > tag_prob) {
            m_n_btag++;
            return eventdata;
        }
    }
    if (b_tag(taup_index)) {
        if (m_rand.get_uniform() > tag_prob) {
            m_n_btag++;
            return eventdata;
        }
    }
    
    // Lepton veto
    if (lepton_veto(10)) {
        m_n_leptonveto++;
        return eventdata;
    }
    
    

    // Get decay products
    int pim_index = 0, pi0m_index = 0, num_index = 0;
    std::vector<int> taum_decay = tauminus.daughterList();
    for (auto itr = taum_decay.begin(); itr != taum_decay.end(); itr++) {
        if (m_pythia.event.at(*itr).id() == -211) pim_index = (*itr);
        if (m_pythia.event.at(*itr).id() == 111) pi0m_index = (*itr);
        if (m_pythia.event.at(*itr).id() == 16) num_index = (*itr);
    }

    int pip_index = 0, pi0p_index = 0, nup_index = 0;
    std::vector<int> taup_decay = tauplus.daughterList();
    for (auto itr = taup_decay.begin(); itr != taup_decay.end(); itr++) {
        if (m_pythia.event.at(*itr).id() == 211) pip_index = (*itr);
        if (m_pythia.event.at(*itr).id() == 111) pi0p_index = (*itr);
        if (m_pythia.event.at(*itr).id() == -16) nup_index = (*itr);
    }

    if (!pim_index || !pi0m_index || !pip_index || !pi0p_index || !num_index || !nup_index) {
        n_tau_decay_errors++;
        if (m_debug) std::cout << "Error with decay products" << std::endl;
        return eventdata;
    }

    // Get the MET
    Vec4 met = sum_met();
    met.pz(0.0);    // this we don't know
    met.e(0.0);

    // 4-vectors for all decay products
    Vec4 piplus = m_pythia.event.at(pip_index).p();
    Vec4 pi0plus = m_pythia.event.at(pi0p_index).p();
    Vec4 nuplus = m_pythia.event.at(nup_index).p();
    Vec4 piminus = m_pythia.event.at(pim_index).p();
    Vec4 pi0minus = m_pythia.event.at(pi0m_index).p();
    Vec4 numinus = m_pythia.event.at(num_index).p();

    Vec4 taup_vis = piplus + pi0plus;
    Vec4 taum_vis = piminus + pi0minus;
    
    // Smear 4-vectors
    if (m_apply_smearing) {

        // Smear tracks
        piplus = smear_fourvector(piplus, 1e-3, 0.05);
        piminus = smear_fourvector(piminus, 1e-3, 0.05);
        
        // Smear pi0's
        // https://arxiv.org/pdf/1512.05955.pdf
        pi0plus = smear_fourvector(pi0plus, 0.025/3.46410, 0.1);
        pi0minus = smear_fourvector(pi0minus, 0.025/3.46410, 0.1);
    }

    // Invariant and transverse mass
    taup_vis = piplus + pi0plus;
    taum_vis = piminus + pi0minus;
    double inv_mass = m(taum_vis, taup_vis);
    double transv_mass = total_transverse_mass(taum_vis, taup_vis, met);
    
    if (inv_mass < m_mass_lower || inv_mass > m_mass_upper) {
        m_n_mass++;
        return eventdata;
    }
    m_n_passed_mass_cuts++;

    
    // Impact parameter vectors
    Vec4 primary_vertex = Vec4(0, 0, 0, 0); // vertex spread is off.
    Vec4 piplus_ipvec = impact_param_vector(m_pythia.event.at(pip_index), primary_vertex);
    Vec4 piminus_ipvec = impact_param_vector(m_pythia.event.at(pim_index), primary_vertex);
    
    
    // Event selection
    float lead_pT = taup_vis.pT();
    float sublead_pT = taum_vis.pT();
    float deltaR = sqrt(pow(taup_vis.eta()-taum_vis.eta(), 2) + pow(taup_vis.phi()-taum_vis.phi(), 2));
    if (lead_pT < sublead_pT) std::swap(lead_pT, sublead_pT);
    
    if (lead_pT < m_min_leading_pt || sublead_pT < m_min_subleading_pt) {
        m_n_pt++;
        return eventdata;
    }
    if (fabs(taup_vis.eta()) > m_max_eta || fabs(taum_vis.eta()) > m_max_eta) {
        m_n_eta++;
        return eventdata;
    }
    if (deltaR < m_min_deltaR) {
        m_n_deltaR++;
        return eventdata;
    }
    if (met.pT() < m_min_MET) {
        m_n_MET++;
        return eventdata;
    }
    
    ++n_accepted_events;
    
    
    
    // Debug output to check the boost/rotations
    if (m_debug) {
        std::cout << "Before rotbst: -----------------------------" << std::endl;
        std::cout << "taup_vis: \t" << taup_vis;
        std::cout << "taum_vis: \t" << taum_vis;
        std::cout << "piplus: \t" << piplus;
        std::cout << "pi0plus: \t" << pi0plus;
        std::cout << "piminus: \t" << piminus;
        std::cout << "pi0minus: \t" << pi0minus;
    }
    
    // Boost back to selected rest frame and rotate so that the visible tau-
    // points in the +z direction
    RotBstMatrix rbm;
    switch (m_restframe) {
        case LAB:
            break;
        case TRUTH:
            rbm.toCMframe(piminus+pi0minus+numinus, piplus+pi0plus+nuplus);
            break;
        case VISIBLE:
            rbm.toCMframe(piminus+pi0minus, piplus+pi0plus);
            break;
        case CHARGED:
            rbm.toCMframe(piminus, piplus);
            break;
    }
    
    taup_vis.rotbst(rbm);
    taum_vis.rotbst(rbm);
    piplus.rotbst(rbm);
    pi0plus.rotbst(rbm);
    piminus.rotbst(rbm);
    pi0minus.rotbst(rbm);
    met.rotbst(rbm);
    piplus_ipvec.rotbst(rbm);
    piminus_ipvec.rotbst(rbm);

    if (m_debug) {
        std::cout << "After rotbst: -----------------------------" << std::endl;
        std::cout << "taup_vis: \t" << taup_vis;
        std::cout << "taum_vis: \t" << taum_vis;
        std::cout << "piplus: \t" << piplus;
        std::cout << "pi0plus: \t" << pi0plus;
        std::cout << "piminus: \t" << piminus;
        std::cout << "pi0minus: \t" << pi0minus;
    }
    
    // Rotate around z so that pi+ is aligned with the x-axis
    rbm.reset();
    double phi = atan2(piplus.py(), piplus.px());
    rbm.rot(0, -phi);
    
    taup_vis.rotbst(rbm);
    taum_vis.rotbst(rbm);
    piplus.rotbst(rbm);
    pi0plus.rotbst(rbm);
    piminus.rotbst(rbm);
    pi0minus.rotbst(rbm);
    met.rotbst(rbm);
    piplus_ipvec.rotbst(rbm);
    piminus_ipvec.rotbst(rbm);
    
    if (m_debug) {
        std::cout << "After x align: -----------------------------" << std::endl;
        std::cout << "taup_vis: \t" << taup_vis;
        std::cout << "taum_vis: \t" << taum_vis;
        std::cout << "piplus: \t" << piplus;
        std::cout << "pi0plus: \t" << pi0plus;
        std::cout << "piminus: \t" << piminus;
        std::cout << "pi0minus: \t" << pi0minus;
        
        std::cout << std::endl << "Continue?" << std::endl;
        char y; std::cin >> y;
    }
    
    // Upsilon and phistar business
    double upsilon_plus = upsilon(piplus, pi0plus);
    double upsilon_minus = upsilon(piminus, pi0minus);
    double triplecorr = acoplanarity(piminus, pi0minus, piplus, pi0plus, true);
    double phi_star = acoplanarity(piminus, pi0minus, piplus, pi0plus, false);
    
    
    // Testing new angles
    double pip_pim_angle = atan2(piminus.py(), piminus.px()); 
    double pi0p_pi0m_angle = atan2(pi0minus.py(), pi0minus.px()); 
    double pip_dot_pi0p = dot3(piplus, pi0plus);
    double pim_dot_pi0m = dot3(piminus, pi0minus);
    double pim_dot_pip = dot3(piminus, piplus);
    double pip_dot_pi0m = dot3(piplus, pi0minus);
    double pim_dot_pi0p = dot3(piminus, pi0plus);
    
    
    // Fill eventdata
    eventdata.push_back(piplus.px());
    eventdata.push_back(piplus.py());
    eventdata.push_back(piplus.pz());
    eventdata.push_back(piplus.e());

    eventdata.push_back(pi0plus.px());
    eventdata.push_back(pi0plus.py());
    eventdata.push_back(pi0plus.pz());
    eventdata.push_back(pi0plus.e());

    eventdata.push_back(piminus.px());
    eventdata.push_back(piminus.py());
    eventdata.push_back(piminus.pz());
    eventdata.push_back(piminus.e());

    eventdata.push_back(pi0minus.px());
    eventdata.push_back(pi0minus.py());
    eventdata.push_back(pi0minus.pz());
    eventdata.push_back(pi0minus.e());

    eventdata.push_back(met.px());
    eventdata.push_back(met.py());

    eventdata.push_back(inv_mass);
    eventdata.push_back(transv_mass);

    eventdata.push_back(upsilon_plus);
    eventdata.push_back(upsilon_minus);
    eventdata.push_back(upsilon_plus*upsilon_minus);
    eventdata.push_back(triplecorr);
    eventdata.push_back(phi_star);
    
    eventdata.push_back(piplus_ipvec.px());
    eventdata.push_back(piplus_ipvec.py());
    eventdata.push_back(piplus_ipvec.pz());
    eventdata.push_back(piminus_ipvec.px());
    eventdata.push_back(piminus_ipvec.py());
    eventdata.push_back(piminus_ipvec.pz());
    
    eventdata.push_back(pip_pim_angle);
    eventdata.push_back(pi0p_pi0m_angle);
    eventdata.push_back(pip_dot_pi0p);
    eventdata.push_back(pim_dot_pi0m);
    eventdata.push_back(pim_dot_pip);
    eventdata.push_back(pip_dot_pi0m);
    eventdata.push_back(pim_dot_pi0p);
    
    // Finally, add target value
    eventdata.push_back(m_target);
    
    
    return eventdata;
}


// -----------------------------------------------------------------------------
int Analysis::scan_past_brem(int ip) {

    int id = m_pythia.event.at(ip).id();
    vector<int> dlist = m_pythia.event.at(ip).daughterList();
    if (dlist.size() == 2) {
        for (int i = 0; i < 2; i++) {
            if (m_pythia.event.at(dlist[i]).id() == id) return scan_past_brem(dlist[i]);
        }
    }
    else if (dlist.size() == 1) {
        if (m_pythia.event.at(dlist[0]).id() == id) return scan_past_brem(dlist[0]);
    }

    return ip;
}

// -----------------------------------------------------------------------------
Vec4 Analysis::unit(const Vec4 vec) {
    Vec4 unitvec(vec);
    unitvec.rescale3(1.0/vec.pAbs());
    return unitvec;
}

// -----------------------------------------------------------------------------
double Analysis::upsilon(const Vec4 charged, const Vec4 neutral) {
    return (charged.e() - neutral.e()) / (charged.e() + neutral.e());
}

// -----------------------------------------------------------------------------
double Analysis::acoplanarity(const Vec4 pim, const Vec4 pi0m, const Vec4 pip,
                              const Vec4 pi0p, bool return_triple_corr) {

    Vec4 piplus3 = Vec4(pip); piplus3.e(0.0);   // -> 3D vector
    Vec4 pi0minus3 = Vec4(pi0m); pi0minus3.e(0.0);
    Vec4 piminus3 = Vec4(pim); piminus3.e(0.0);
    Vec4 pi0plus3 = Vec4(pi0p); pi0plus3.e(0.0);

    // Parallel projections of pi0 onto pi+/- direction
    double dotprod_plus = dot3(pi0plus3, unit(piplus3));
    Vec4 neutral_parallel_plus = dotprod_plus * unit(piplus3);

    double dotprod_minus = dot3(pi0minus3, unit(piminus3));
    Vec4 neutral_parallel_minus = dotprod_minus * unit(piminus3);

    // Perpendicular component of pi0
    Vec4 neutral_perp_plus = unit(pi0plus3 - neutral_parallel_plus);
    Vec4 neutral_perp_minus = unit(pi0minus3 - neutral_parallel_minus);

    // Acoplanarity angle, [0, pi)
    double phistar = acos(dot3(neutral_perp_plus, neutral_perp_minus));

    // Use T-odd triple correlation product to get angle between 0 and 2pi
    double triplecorr = dot3(piminus3, cross3(neutral_perp_plus, neutral_perp_minus));
    if (return_triple_corr) {
        return triplecorr;
    }
        
    if (triplecorr < 0) {
        phistar = 2*M_PI - phistar;
    }

    // Upsilon separation
    double y0y1 = upsilon(pip, pi0p)*upsilon(pim, pi0m);
    if (y0y1 < 0) {
        if (phistar > M_PI) phistar -= M_PI;
        else phistar += M_PI;
    }
    return phistar;
}


// -----------------------------------------------------------------------------
Vec4 Analysis::impact_param_vector(const Particle pion, const Vec4 privertex) {
    
    Vec4 pionvector = Vec4(pion.p());
    pionvector.e(0.0);  // Create 3-vector
    Vec4 tauvertex = pion.vProd();  // Tau decay vertex 
    
    // Determine shortest distance from primary vertex to the extrapolated pion track
    double k = dot3(privertex - tauvertex, pionvector) / pionvector.pAbs2();    
    
    Vec4 ipvector = tauvertex + k*pionvector - privertex;
    return ipvector;
}


// -----------------------------------------------------------------------------
Vec4 Analysis::sum_met() {

    Vec4 met4;

    for (int i = 0; i < m_pythia.event.size(); i++) {
        // Tau neutrinos
        if (abs(m_pythia.event.at(i).id()) == 16) {
            met4 += m_pythia.event.at(i).p();
        }
        // ...add others too?
    }
    return met4;
}

// -----------------------------------------------------------------------------
double Analysis::total_transverse_mass(const Vec4 tau1, const Vec4 tau2, const Vec4 met) {

    auto mT = [](const Vec4 &a, const Vec4 &b) -> double {
        return 2*a.pT()*b.pT()*(1-cos(a.phi()-b.phi()));
    };

    return sqrt(mT(met, tau1) + mT(met, tau2) + mT(tau1, tau2));
}


// -----------------------------------------------------------------------------
bool Analysis::b_tag(const int tauindex) {
    

    // Lambda for iterating up to the top particle in the chain with same pdg id
    // (for some reason iTopCopyId doesn't work)
    std::function<int(int)> traverse_up;
    traverse_up = [&traverse_up, this](const int index) -> int {
        
        int id = m_pythia.event.at(index).id();
        
        int mother1 = m_pythia.event.at(index).mother1();
        if (m_pythia.event.at(mother1).id() == id) return traverse_up(mother1);
        
        int mother2 = m_pythia.event.at(index).mother2();
        if (m_pythia.event.at(mother2).id() == id) return traverse_up(mother2);
        
        return index;
    };
    
    // Get the upper tau index
    int firstindex = traverse_up(tauindex);
    
    // Same for mother
    int motherindex = m_pythia.event.at(firstindex).mother1();
    Particle mother = m_pythia.event.at(traverse_up(motherindex));
    
    if (mother.idAbs() == 24) { // Mother is a W
        
        Particle grandmother = m_pythia.event.at(mother.mother1());
        
        if (grandmother.idAbs() == 6) { // Grandmother is a top quark
            
            // Check if a b-quark is present in top decay (should be)
            if ((m_pythia.event.at(grandmother.daughter1()).idAbs() == 5) ||
                (m_pythia.event.at(grandmother.daughter2()).idAbs() == 5)) {
                    return true;
                }
        }
    }

    return false;
}




// -----------------------------------------------------------------------------
bool Analysis::lepton_veto(float minpt) {
    
    size_t n_taus = 0;
    std::vector<int> daughters;
    bool taucopy;
    
    for (int i = 0; i < m_pythia.event.size(); i++) {
        
        // Reject e/mu
        if (m_pythia.event.at(i).idAbs() == 11 || m_pythia.event.at(i).idAbs() == 13) {
            if (m_pythia.event.at(i).pT() > minpt) return true;
        }
        
        // Reject if too many taus
        if (m_pythia.event.at(i).idAbs() == 15) {
            
            // Check that it's a decaying tau
            taucopy = false;
            daughters = m_pythia.event.at(i).daughterList();
            for (size_t j = 0; j < daughters.size(); j++) {
                if (m_pythia.event.at(daughters.at(j)).idAbs() == 15) taucopy = true;
            }
            if (!taucopy and m_pythia.event.at(i).pT() > minpt) n_taus++;
            if (n_taus > 2) return true;
        }
    }
    
    return false;
}


// -----------------------------------------------------------------------------
Pythia8::Vec4 Analysis::smear_fourvector(Pythia8::Vec4 pin, float angular_res, float energy_res) {
    
    // Unit vector of input
    Pythia8::Vec4 pin_unit(pin);
    pin_unit.rescale3(1.0/pin_unit.pAbs());

    // Unit vector in z direction 
    Pythia8::Vec4 z(0, 0, 1, 0);

    // Find a random unit vector perpendicular to the z axis
    float randangle = m_rand.get_uniform()*2*M_PI;
    Pythia8::Vec4 smearaxis(cos(randangle), sin(randangle), 0, 0);
    smearaxis.rescale3(1.0/smearaxis.pAbs());

    // Rotate it, so it becomes prependicular to pin
    Pythia8::Vec4 rotaxis = cross3(pin_unit, z);
    float rotangle = -acos(dot3(pin_unit, z));
    smearaxis.rotaxis(rotangle, rotaxis);

    // Now smear output vector by random angle around the smearing axis (which is also random)
    Pythia8::Vec4 pout(pin);
    pout.rotaxis(m_rand.get_normal()*angular_res, smearaxis);

    // Smear the total energy
    // Offset the random number so that it is centered at 1, not 0
    pout.rescale3((m_rand.get_normal() * energy_res) + 1);

    // Retain the original mass
    double m2 = pin.m2Calc();

    // Recompute energy
    pout.p(pout.px(), pout.py(), pout.pz(),
           sqrt(pout.px()*pout.px() + 
                pout.py()*pout.py() + 
                pout.pz()*pout.pz() + m2
            )
          );

    return pout;

}

