from gillespie_simple import *

from pathlib import Path

# Concentrations
conc_init_prey = 4 # M
conc_init_hunter = 10 # M

# Fake volume
vol = 10**(-21) # L

# Initial counts
no_init_prey = int(conc_init_prey * vol * AVOGADRO)
no_init_hunter = int(conc_init_hunter * vol * AVOGADRO)
print("Initial no prey: %d" % no_init_prey)
print("Initial no hunter: %d" % no_init_hunter)

# Reaction
rxn1 = Rxn("1",1.0,["P"],["P","P"])
rxn2 = Rxn("2",0.1,["H"],[])

conc_based_rate = 0.1
count_based_rate = conc_based_rate / (vol * AVOGADRO)
rxn3 = Rxn("3",count_based_rate,["H","P"],["H","H"])

no_seeds = 25
for seed in range(0,no_seeds):
    print("Seed: %d / %d" % (seed,no_seeds))

    # Initial counts
    counts = Counts()
    counts.set_count("H",no_init_hunter)
    counts.set_count("P",no_init_prey)

    # Run
    counts_hist = run_gillespie(
        rxn_list=[rxn1,rxn2,rxn3],
        counts=counts,
        dt_st_every=0.1,
        t_max=50,
        verbose=False
        )

    # Write
    Path("data").mkdir(parents=True, exist_ok=True)
    counts_hist.write_all_count_hists_single_file("data/%02d.txt" % seed)
