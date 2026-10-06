"""Self-check questions for entrance exam units, graded on the server.

Why only here. The weekly course path is built on the Open University data, whose course
content was anonymised: a week there is a position in a course, not a topic, so a
question about it would be invented and its answer would feed the model evidence about
something the student never studied. Entrance exam units are real syllabus topics, so a
question about Kinematics is a question about Kinematics.

What a check is, and is not. Three questions per topic, written to have exactly one right
answer, each with the reason. They tell a student whether a unit has stuck; they are not
a mock paper and do not predict a score. Results are kept apart from the study events the
model reads, because an exam unit is not a week of a course.

Options are stored with the right answer first and shuffled on the way out, the same way
every time for the same question, so the right answer is not always in the same place
and a page reload does not move it.

Units map to topics by (section, unit) rather than by unit name: NEET has a
"Thermodynamics" unit under Physics and another under Chemistry.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from elpr import exams

PASS_MARK = 40          # the same line as a typed quiz mark (sql/05_events.sql)


@dataclass(frozen=True)
class Question:
    prompt: str
    right: str
    wrong: tuple        # three wrong options
    why: str


def Q(prompt: str, right: str, wrong: tuple, why: str) -> Question:
    return Question(prompt, right, tuple(wrong), why)


BANK: dict[str, tuple[Question, ...]] = {
    # ------------------------------------------------------------------ physics
    "measurement": (
        Q("Which of these is an SI base unit?", "kelvin", ("newton", "joule", "watt"),
          "The seven base units are metre, kilogram, second, ampere, kelvin, mole and "
          "candela. The newton, joule and watt are built from them."),
        Q("What are the dimensions of force?", "[M L T⁻²]",
          ("[M L T⁻¹]", "[M L² T⁻²]", "[M L⁻¹ T⁻²]"),
          "Force is mass times acceleration, and acceleration is length over time squared."),
        Q("How many significant figures are there in 0.00450?", "3", ("2", "5", "6"),
          "Leading zeros never count. The trailing zero after the decimal point does, so "
          "the figures are 4, 5 and 0."),
    ),
    "kinematics": (
        Q("A body starts from rest and accelerates uniformly at 2 m/s². How far does it "
          "travel in 5 s?", "25 m", ("10 m", "20 m", "50 m"),
          "s = ½at² = ½ × 2 × 5² = 25 m."),
        Q("On level ground with no air resistance, which launch angle gives a projectile "
          "the greatest range?", "45°", ("30°", "60°", "90°"),
          "Range is u² sin 2θ / g, largest when 2θ = 90°."),
        Q("The slope of a velocity-time graph gives the", "acceleration",
          ("displacement", "distance travelled", "speed"),
          "Slope is change in velocity over change in time, which is acceleration."),
    ),
    "laws_of_motion": (
        Q("A net force of 10 N acts on a 2 kg mass. What is its acceleration?", "5 m/s²",
          ("0.2 m/s²", "12 m/s²", "20 m/s²"), "a = F/m = 10/2 = 5 m/s²."),
        Q("Which law explains why a gun recoils when it is fired?", "Newton's third law",
          ("Newton's first law", "Newton's second law", "Hooke's law"),
          "The bullet is pushed forward, so the gun is pushed back with an equal and "
          "opposite force."),
        Q("Under the usual laws of friction, limiting friction does not depend on the",
          "area of contact", ("normal reaction", "nature of the two surfaces",
                              "coefficient of friction"),
          "Limiting friction is the coefficient of friction times the normal reaction; "
          "the area in contact does not appear."),
    ),
    "work_energy": (
        Q("A 2 kg ball moves at 3 m/s. What is its kinetic energy?", "9 J",
          ("3 J", "6 J", "18 J"), "KE = ½mv² = ½ × 2 × 3² = 9 J."),
        Q("The SI unit of power is the", "watt", ("joule", "newton", "pascal"),
          "Power is work per unit time, one joule per second being one watt."),
        Q("In a perfectly elastic collision, what is conserved?",
          "Both momentum and kinetic energy",
          ("Only momentum", "Only kinetic energy", "Neither"),
          "Momentum is conserved in every collision; elastic means kinetic energy is "
          "conserved as well."),
    ),
    "rotation": (
        Q("The rotational counterpart of mass is", "moment of inertia",
          ("torque", "angular momentum", "angular velocity"),
          "Moment of inertia measures resistance to angular acceleration, as mass does "
          "for linear acceleration."),
        Q("When no external torque acts on a system, which quantity stays constant?",
          "Angular momentum", ("Angular acceleration", "Torque", "Rotational kinetic energy"),
          "Torque is the rate of change of angular momentum, so zero torque leaves it "
          "unchanged. Kinetic energy can still change, as when a skater pulls their arms in."),
        Q("The moment of inertia of a thin ring of mass M and radius R about the axis "
          "through its centre, perpendicular to its plane, is", "MR²",
          ("½MR²", "⅔MR²", "⅖MR²"),
          "Every bit of the ring sits at distance R from the axis."),
    ),
    "gravitation": (
        Q("If the distance between two masses is doubled, the gravitational force "
          "between them becomes", "one quarter as large",
          ("half as large", "twice as large", "four times as large"),
          "Gravity follows an inverse square law: (1/2)² = 1/4."),
        Q("The escape velocity from the Earth's surface is about", "11.2 km/s",
          ("7.9 km/s", "3.0 km/s", "42 km/s"),
          "√(2gR) with g ≈ 9.8 m/s² and R ≈ 6400 km gives about 11.2 km/s. 7.9 km/s is "
          "the orbital speed near the surface."),
        Q("Kepler's second law, equal areas in equal times, follows from conservation of",
          "angular momentum", ("energy", "linear momentum", "mass"),
          "Gravity acts along the line to the Sun, so it exerts no torque about the Sun."),
    ),
    "bulk_matter": (
        Q("Young's modulus is the ratio of",
          "longitudinal stress to longitudinal strain",
          ("shear stress to shear strain", "volume stress to volume strain",
           "longitudinal strain to longitudinal stress"),
          "Young's modulus describes stretching along a length. Shear and bulk modulus "
          "are the other two."),
        Q("Why does a small drop of liquid take a spherical shape?", "Surface tension",
          ("Gravity", "Viscosity", "Buoyancy"),
          "Surface tension minimises surface area, and a sphere has the least area for a "
          "given volume."),
        Q("Bernoulli's principle for a flowing fluid is a statement of conservation of",
          "energy", ("mass", "momentum", "charge"),
          "It balances pressure energy, kinetic energy and potential energy per unit "
          "volume along a streamline."),
    ),
    "phys_thermo": (
        Q("In an isothermal process for an ideal gas, which quantity stays constant?",
          "Temperature", ("Pressure", "Volume", "Heat supplied"),
          "Iso-thermal means equal temperature."),
        Q("The first law of thermodynamics is a statement of conservation of", "energy",
          ("momentum", "entropy", "mass"),
          "Heat supplied equals the rise in internal energy plus the work done."),
        Q("What is the efficiency of a Carnot engine working between 500 K and 300 K?",
          "40%", ("60%", "20%", "67%"), "Efficiency = 1 − T₂/T₁ = 1 − 300/500 = 0.4."),
    ),
    "kinetic_theory": (
        Q("By kinetic theory, the average kinetic energy of a gas molecule is directly "
          "proportional to its", "absolute temperature", ("pressure", "volume", "density"),
          "Average translational KE = (3/2)kT."),
        Q("Which is the ideal gas equation?", "PV = nRT",
          ("PV = RT/n", "P = nRV", "PT = nRV"),
          "Pressure times volume equals moles times the gas constant times temperature."),
        Q("How many degrees of freedom does a molecule of a monatomic gas have?", "3",
          ("1", "5", "6"),
          "It can only move along three directions; a single atom has no rotation that "
          "counts."),
    ),
    "oscillations_waves": (
        Q("For small swings, the time period of a simple pendulum depends on", "its length",
          ("the mass of the bob", "the material of the bob", "the size of the swing"),
          "T = 2π√(l/g): only length and g appear."),
        Q("Sound waves in air are", "longitudinal",
          ("transverse", "electromagnetic", "unable to reflect"),
          "Air molecules vibrate along the direction the sound travels."),
        Q("A wave has a frequency of 50 Hz and a wavelength of 2 m. What is its speed?",
          "100 m/s", ("25 m/s", "52 m/s", "0.04 m/s"), "v = fλ = 50 × 2 = 100 m/s."),
    ),
    "electrostatics": (
        Q("If the distance between two point charges is halved, the force between them "
          "becomes", "four times as large",
          ("half as large", "twice as large", "one quarter as large"),
          "Coulomb's law is inverse square: 1/(1/2)² = 4."),
        Q("Inside a charged conductor in electrostatic equilibrium, the electric field is",
          "zero", ("largest at the centre", "equal to the field at the surface", "infinite"),
          "Free charges move until there is no field left inside to push them."),
        Q("The SI unit of capacitance is the", "farad", ("coulomb", "henry", "volt"),
          "One farad stores one coulomb per volt."),
    ),
    "current_electricity": (
        Q("Three 6 Ω resistors are connected in parallel. What is the equivalent "
          "resistance?", "2 Ω", ("18 Ω", "3 Ω", "6 Ω"), "1/R = 1/6 + 1/6 + 1/6 = 1/2."),
        Q("Kirchhoff's junction rule is based on conservation of", "charge",
          ("energy", "momentum", "mass"),
          "Charge cannot pile up at a junction, so current in equals current out."),
        Q("A 12 V supply drives 3 A through a resistor. What is its resistance?", "4 Ω",
          ("36 Ω", "0.25 Ω", "15 Ω"), "R = V/I = 12/3 = 4 Ω."),
    ),
    "magnetism": (
        Q("A charged particle moves parallel to a uniform magnetic field. The magnetic "
          "force on it is", "zero",
          ("at its largest", "along the field", "opposite to the field"),
          "F = qvB sin θ, and θ = 0 for motion along the field."),
        Q("The SI unit of magnetic flux density is the", "tesla",
          ("weber", "henry", "gauss"),
          "The weber is the unit of flux; the gauss is the older CGS unit."),
        Q("A moving coil galvanometer is turned into an ammeter by connecting",
          "a low resistance in parallel",
          ("a high resistance in series", "a low resistance in series",
           "a high resistance in parallel"),
          "The low resistance shunt carries most of the current around the coil."),
    ),
    "emi_ac": (
        Q("Lenz's law is a consequence of conservation of", "energy",
          ("charge", "momentum", "mass"),
          "If the induced current helped the change that caused it, energy would appear "
          "from nothing."),
        Q("A transformer works only with", "alternating current",
          ("direct current", "a permanent magnet", "a battery"),
          "It needs a changing magnetic flux, which a steady current does not make."),
        Q("The SI unit of self-inductance is the", "henry", ("tesla", "farad", "weber"),
          "Named after Joseph Henry."),
    ),
    "em_waves": (
        Q("Which of these has the longest wavelength?", "Radio waves",
          ("X-rays", "Visible light", "Microwaves"),
          "Radio waves sit at the long wavelength, low frequency end of the spectrum."),
        Q("Electromagnetic waves are", "transverse",
          ("longitudinal", "possible only in a medium", "slower in vacuum than in glass"),
          "The electric and magnetic fields vibrate at right angles to the direction of "
          "travel."),
        Q("In vacuum, every electromagnetic wave travels at about", "3 × 10⁸ m/s",
          ("340 m/s", "3 × 10⁶ m/s", "a speed that depends on its frequency"),
          "All of them move at the speed of light in vacuum."),
    ),
    "optics": (
        Q("A concave mirror has a focal length of 10 cm. What is its radius of curvature?",
          "20 cm", ("5 cm", "10 cm", "40 cm"), "R = 2f."),
        Q("Total internal reflection can happen when light travels from",
          "glass into air", ("air into glass", "air into water", "vacuum into water"),
          "It needs light going from a denser medium into a rarer one."),
        Q("What is the power of a convex lens with a focal length of 50 cm?", "+2 D",
          ("+0.5 D", "−2 D", "+50 D"),
          "P = 1/f with f in metres: 1/0.5 = 2 dioptres, positive for a convex lens."),
    ),
    "dual_nature": (
        Q("In the photoelectric effect, the maximum kinetic energy of the emitted "
          "electrons depends on the", "frequency of the light",
          ("intensity of the light", "time the light is on", "area of the plate"),
          "Each photon gives hf; brighter light gives more photons, not more energetic ones."),
        Q("The de Broglie wavelength of a particle with momentum p is", "h/p",
          ("p/h", "hp", "h/p²"), "λ = h/p."),
        Q("Light below the threshold frequency falls on a metal. What happens?",
          "No electrons are emitted, however bright the light",
          ("Electrons are emitted after a delay",
           "Electrons are emitted if the light is bright enough",
           "Electrons are emitted only in a vacuum"),
          "Each photon must carry at least the work function; adding photons does not help."),
    ),
    "atoms_nuclei": (
        Q("In Bohr's model, an electron's angular momentum in an allowed orbit is a whole "
          "number multiple of", "h/2π", ("h", "2πh", "h/π"), "mvr = nh/2π."),
        Q("A radioactive sample has a half-life of 10 days. What fraction is left after "
          "30 days?", "1/8", ("1/3", "1/4", "1/6"),
          "30 days is three half-lives: (1/2)³ = 1/8."),
        Q("Why does nuclear fission release energy?",
          "The products have a higher binding energy per nucleon",
          ("Mass is created in the reaction", "Electrons are released",
           "The nucleus gains protons"),
          "The more tightly bound products have slightly less mass, and the difference is "
          "released as energy."),
    ),
    "electronic_devices": (
        Q("In an n-type semiconductor the majority charge carriers are", "electrons",
          ("holes", "protons", "positive ions"),
          "The donor impurity contributes extra electrons."),
        Q("A p-n junction diode conducts easily when it is", "forward biased",
          ("reverse biased", "left unbiased", "cooled to absolute zero"),
          "Forward bias narrows the depletion layer and lets current flow."),
        Q("The output of an AND gate is 1 only when", "all of its inputs are 1",
          ("any input is 1", "all of its inputs are 0", "its inputs differ"),
          "AND needs every input to be true."),
    ),
    "experimental": (
        Q("A vernier calliper has 10 vernier divisions equal to 9 main scale divisions of "
          "1 mm. What is its least count?", "0.1 mm", ("0.01 mm", "0.9 mm", "1 mm"),
          "One main division minus one vernier division: 1 − 0.9 = 0.1 mm."),
        Q("A screw gauge has a pitch of 0.5 mm and 50 divisions on its circular scale. "
          "What is its least count?", "0.01 mm", ("0.05 mm", "0.1 mm", "0.5 mm"),
          "Least count = pitch / divisions = 0.5/50 = 0.01 mm."),
        Q("In a metre bridge experiment, the balance point is where",
          "the galvanometer shows no deflection",
          ("the galvanometer shows full deflection", "the cell gives no current",
           "the two resistances are equal"),
          "At balance no current flows through the galvanometer."),
    ),
    # ---------------------------------------------------------------- chemistry
    "basic_concepts": (
        Q("How many moles are there in 36 g of water? (molar mass 18 g/mol)", "2",
          ("1", "18", "0.5"), "n = mass / molar mass = 36/18 = 2."),
        Q("Avogadro's number is about", "6.022 × 10²³",
          ("3.0 × 10⁸", "1.6 × 10⁻¹⁹", "6.626 × 10⁻³⁴"),
          "The others are the speed of light, the electron's charge and Planck's constant."),
        Q("0.5 mol of a solute is dissolved to make 250 mL of solution. What is the "
          "molarity?", "2 M", ("0.125 M", "0.5 M", "1 M"),
          "Molarity = moles per litre = 0.5 / 0.25 = 2 M."),
    ),
    "atomic_structure": (
        Q("What is the largest number of electrons the n = 3 shell can hold?", "18",
          ("8", "9", "32"), "A shell holds 2n² electrons: 2 × 3² = 18."),
        Q("Which quantum number describes the shape of an orbital?",
          "The azimuthal quantum number (l)",
          ("The principal quantum number (n)", "The magnetic quantum number (m)",
           "The spin quantum number (s)"),
          "l = 0, 1, 2, 3 gives s, p, d and f shapes."),
        Q("Hund's rule says that electrons entering orbitals of equal energy",
          "fill them singly with parallel spins before pairing",
          ("pair up before any orbital is singly filled",
           "all have the same four quantum numbers",
           "go into the highest energy orbital first"),
          "Unpaired parallel spins give the lowest energy arrangement."),
    ),
    "bonding": (
        Q("What is the shape of a methane (CH₄) molecule?", "Tetrahedral",
          ("Square planar", "Trigonal planar", "Linear"),
          "Four bond pairs around carbon, sp³ hybridised, at about 109.5°."),
        Q("The H-O-H bond angle in water is about", "104.5°",
          ("180°", "120°", "109.5°"),
          "Two lone pairs on oxygen push the bonds closer than the tetrahedral 109.5°."),
        Q("How many sigma and pi bonds are there in a nitrogen molecule, N≡N?",
          "1 sigma and 2 pi", ("3 sigma", "2 sigma and 1 pi", "3 pi"),
          "Every triple bond is one sigma bond and two pi bonds."),
    ),
    "chem_thermo": (
        Q("At constant temperature and pressure, a process is spontaneous when", "ΔG < 0",
          ("ΔG > 0", "ΔH > 0", "ΔS < 0"),
          "A negative Gibbs energy change is the test for spontaneity."),
        Q("Hess's law holds because enthalpy is", "a state function",
          ("a path function", "always negative", "equal to the work done"),
          "The change depends only on where you start and finish, not on the route."),
        Q("For an exothermic reaction, ΔH is", "negative",
          ("positive", "zero", "always larger than ΔG"),
          "Heat leaves the system, so its enthalpy falls."),
    ),
    "solutions": (
        Q("Which of these is a colligative property?", "Elevation of boiling point",
          ("Colour of the solution", "Density", "Refractive index"),
          "Colligative properties depend only on the number of solute particles."),
        Q("Raoult's law relates the vapour pressure of a component to its",
          "mole fraction in the solution", ("mass percentage", "density", "normality"),
          "p = p° × x for each component of an ideal solution."),
        Q("Assuming complete dissociation, the van 't Hoff factor of NaCl is", "2",
          ("1", "3", "0.5"), "Each formula unit gives two ions, Na⁺ and Cl⁻."),
    ),
    "equilibrium": (
        Q("What is the pH of a 0.01 M solution of HCl?", "2", ("1", "12", "0.01"),
          "HCl is a strong acid: [H⁺] = 10⁻², so pH = 2."),
        Q("Adding a catalyst to a reaction at equilibrium",
          "does not change the equilibrium position",
          ("shifts it towards the products", "shifts it towards the reactants",
           "changes the equilibrium constant"),
          "A catalyst speeds up the forward and backward reactions equally."),
        Q("For N₂ + 3H₂ ⇌ 2NH₃, raising the pressure shifts the equilibrium",
          "towards ammonia", ("towards nitrogen and hydrogen", "nowhere",
                              "in a direction that depends on the catalyst"),
          "Four moles of gas become two, so higher pressure favours the side with fewer."),
    ),
    "redox": (
        Q("Oxidation is", "loss of electrons",
          ("gain of electrons", "gain of protons", "loss of neutrons"),
          "OIL RIG: oxidation is loss, reduction is gain."),
        Q("What is the oxidation number of manganese in KMnO₄?", "+7",
          ("+2", "+4", "+6"), "+1 for K and 4 × (−2) for O leave +7 for Mn."),
        Q("In Zn + Cu²⁺ → Zn²⁺ + Cu, the reducing agent is", "Zn",
          ("Cu²⁺", "Zn²⁺", "Cu"), "Zinc gives up electrons, reducing the copper ion."),
    ),
    "electrochem": (
        Q("In a galvanic cell, oxidation takes place at the", "anode",
          ("cathode", "salt bridge", "both electrodes equally"),
          "An Ox, Red Cat: oxidation at the anode, reduction at the cathode."),
        Q("The standard hydrogen electrode is given a potential of", "0 V",
          ("1 V", "−1 V", "1.23 V"),
          "Every other standard electrode potential is measured against it."),
        Q("Faraday's constant, the charge on one mole of electrons, is about",
          "96,500 C", ("1.6 × 10⁻¹⁹ C", "6.022 × 10²³ C", "8.314 C"),
          "Avogadro's number times the charge on one electron."),
    ),
    "kinetics": (
        Q("The unit of the rate constant for a first-order reaction is", "s⁻¹",
          ("mol L⁻¹ s⁻¹", "L mol⁻¹ s⁻¹", "it has no unit"),
          "Rate = k[A], so k has the unit of rate divided by concentration."),
        Q("The half-life of a first-order reaction",
          "does not depend on the starting concentration",
          ("depends on the starting concentration", "doubles as the reaction goes on",
           "is zero"),
          "t½ = 0.693/k, with no concentration in it."),
        Q("A catalyst speeds up a reaction by", "lowering the activation energy",
          ("raising the temperature", "making ΔH larger", "shifting the equilibrium"),
          "It offers a route with a lower energy barrier."),
    ),
    "periodicity": (
        Q("Across a period from left to right, atomic radius generally", "decreases",
          ("increases", "stays the same", "rises and then falls"),
          "Nuclear charge grows while electrons go into the same shell, pulling it in."),
        Q("Which element has the highest electronegativity?", "Fluorine",
          ("Oxygen", "Chlorine", "Nitrogen"), "Fluorine tops the Pauling scale at 4.0."),
        Q("The modern periodic table orders the elements by increasing", "atomic number",
          ("atomic mass", "number of neutrons", "density"),
          "Moseley showed atomic number is the real basis of periodicity."),
    ),
    "p_block": (
        Q("Which noble gas is the most plentiful in the Earth's atmosphere?", "Argon",
          ("Helium", "Neon", "Xenon"), "Argon makes up nearly 1% of air."),
        Q("In their compounds, the halogens most commonly show an oxidation state of",
          "−1", ("+1", "+7", "0"),
          "One electron short of a noble gas configuration, they most often gain one."),
        Q("Ozone (O₃) is an allotrope of", "oxygen", ("nitrogen", "carbon", "sulphur"),
          "Ozone and O₂ are two forms of the same element."),
    ),
    "d_f_block": (
        Q("Compounds of transition metals are often coloured because of",
          "electron transitions between d orbitals",
          ("nuclear reactions", "their s electrons", "their high density"),
          "Split d orbitals absorb part of visible light; we see the rest."),
        Q("The lanthanoid contraction is",
          "a steady decrease in atomic and ionic radii across the lanthanoids",
          ("an increase in radius across the lanthanoids",
           "the shrinking of the nucleus on heating",
           "the loss of f electrons in water"),
          "4f electrons shield the growing nuclear charge poorly."),
        Q("Which of these ions is colourless in water?", "Zn²⁺",
          ("Cu²⁺", "Ni²⁺", "Fe³⁺"),
          "Zn²⁺ has a full d¹⁰ set, so no d-d transition is possible."),
    ),
    "coordination": (
        Q("What is the coordination number of cobalt in [Co(NH₃)₆]³⁺?", "6",
          ("3", "4", "9"), "Six ammonia ligands, each bonding once."),
        Q("Ethylenediamine (en) is a ligand that is", "bidentate",
          ("monodentate", "tridentate", "hexadentate"),
          "Its two nitrogen atoms both bond to the metal."),
        Q("What is the oxidation state of iron in [Fe(CN)₆]⁴⁻?", "+2",
          ("+3", "+4", "+6"), "x + 6 × (−1) = −4, so x = +2."),
    ),
    "purification": (
        Q("Which method suits a liquid that decomposes at its boiling point?",
          "Distillation under reduced pressure",
          ("Simple distillation", "Sublimation", "Crystallisation"),
          "Lower pressure lowers the boiling point below where it decomposes."),
        Q("Lassaigne's test detects", "nitrogen, sulphur and halogens in an organic compound",
          ("the molar mass of a compound", "its melting point", "its optical activity"),
          "Fusing with sodium turns them into ions that ordinary tests can find."),
        Q("Chromatography separates substances because they",
          "distribute differently between a stationary and a mobile phase",
          ("have different colours", "boil at different temperatures",
           "have different masses"),
          "Each substance moves at its own rate depending on how strongly it is held."),
    ),
    "goc": (
        Q("Which carbocation is the most stable?", "Tertiary",
          ("Methyl", "Primary", "Secondary"),
          "More alkyl groups mean more hyperconjugation and inductive release."),
        Q("A nucleophile is a species that", "donates a pair of electrons",
          ("accepts a pair of electrons", "is always positively charged",
           "has no lone pair"),
          "Nucleophile means nucleus-loving: it seeks an electron-poor centre."),
        Q("What is the IUPAC name of CH₃CH₂OH?", "Ethanol",
          ("Methanol", "Propanol", "Ethanal"),
          "Two carbons (eth-) with an -OH group (-ol)."),
    ),
    "hydrocarbons": (
        Q("The general formula of the alkanes is", "CₙH₂ₙ₊₂",
          ("CₙH₂ₙ", "CₙH₂ₙ₋₂", "CₙHₙ"),
          "CₙH₂ₙ is for alkenes and CₙH₂ₙ₋₂ for alkynes."),
        Q("Benzene typically undergoes", "electrophilic substitution",
          ("nucleophilic addition", "free radical polymerisation", "elimination"),
          "Substitution keeps the stable aromatic ring intact."),
        Q("By Markovnikov's rule, HBr adds to propene to give mainly", "2-bromopropane",
          ("1-bromopropane", "propane", "1,2-dibromopropane"),
          "Hydrogen goes to the carbon that already has more hydrogens."),
    ),
    "halogens": (
        Q("An SN2 reaction at a chiral carbon proceeds with", "inversion of configuration",
          ("retention of configuration", "complete racemisation", "no change at all"),
          "The nucleophile attacks from the side opposite the leaving group."),
        Q("Which reacts fastest by the SN1 mechanism?", "(CH₃)₃CCl",
          ("CH₃Cl", "CH₃CH₂Cl", "(CH₃)₂CHCl"),
          "It forms a tertiary carbocation, the most stable of the four."),
        Q("What is the formula of chloroform?", "CHCl₃",
          ("CH₃Cl", "CH₂Cl₂", "CCl₄"), "Trichloromethane."),
    ),
    "alcohols": (
        Q("Phenol is more acidic than ethanol because",
          "the phenoxide ion is stabilised by resonance",
          ("phenol has more hydrogen atoms", "phenol is a gas",
           "phenol has a higher molar mass"),
          "The negative charge spreads over the benzene ring."),
        Q("Lucas reagent is", "concentrated HCl with anhydrous ZnCl₂",
          ("concentrated H₂SO₄ with HNO₃", "NaOH with iodine", "KMnO₄ with H₂SO₄"),
          "It tells primary, secondary and tertiary alcohols apart by how fast they turn "
          "cloudy."),
        Q("Williamson synthesis is used to make", "ethers",
          ("esters", "aldehydes", "amines"),
          "An alkoxide reacts with an alkyl halide."),
    ),
    "carbonyls": (
        Q("Which of these gives a silver mirror with Tollens' reagent?", "Ethanal",
          ("Propanone", "Ethanol", "Ethanoic acid"),
          "Aldehydes are oxidised by Tollens' reagent; ketones are not."),
        Q("Which of these gives a positive iodoform test?", "Propanone",
          ("Methanal", "Benzaldehyde", "Methanoic acid"),
          "The test needs a CH₃CO- group, which propanone has."),
        Q("Reducing an aldehyde gives", "a primary alcohol",
          ("a secondary alcohol", "a tertiary alcohol", "a ketone"),
          "The -CHO carbon becomes -CH₂OH."),
    ),
    "amines": (
        Q("Hinsberg's reagent, used to tell amines apart, is", "benzenesulphonyl chloride",
          ("acetyl chloride", "nitrous acid", "a Grignard reagent"),
          "Primary, secondary and tertiary amines react with it differently."),
        Q("Which test is given only by primary amines?", "The carbylamine test",
          ("The iodoform test", "Tollens' test", "The Lucas test"),
          "With chloroform and alkali they give a foul-smelling isocyanide."),
        Q("Aniline is a weaker base than methylamine because",
          "its nitrogen lone pair is spread into the benzene ring",
          ("aniline has more hydrogen atoms", "methylamine is aromatic",
           "aniline is a gas"),
          "A delocalised lone pair is less available to accept a proton."),
    ),
    "biomolecules": (
        Q("Glucose is", "an aldohexose", ("a ketohexose", "an aldopentose", "a disaccharide"),
          "Six carbons with an aldehyde group."),
        Q("Amino acids in a protein are joined by", "peptide bonds",
          ("glycosidic bonds", "phosphodiester bonds", "hydrogen bonds"),
          "A -CO-NH- link forms between neighbouring amino acids."),
        Q("Which base is found in RNA but not in DNA?", "Uracil",
          ("Thymine", "Adenine", "Guanine"), "RNA uses uracil where DNA uses thymine."),
    ),
    "practical": (
        Q("A suitable indicator for titrating a strong acid against a strong base is",
          "phenolphthalein", ("starch", "Fehling's solution", "Lucas reagent"),
          "It changes colour within the steep pH jump at the end point."),
        Q("In salt analysis, a brick-red flame points to", "calcium",
          ("sodium", "potassium", "copper"),
          "Sodium is golden yellow, potassium lilac, copper blue-green."),
        Q("A white precipitate with silver nitrate that dissolves in ammonia solution "
          "points to", "chloride", ("bromide", "iodide", "sulphate"),
          "AgCl is white and soluble in ammonia; AgBr and AgI are cream and yellow."),
    ),
    # -------------------------------------------------------------- mathematics
    "sets": (
        Q("A set has 3 elements. How many subsets does it have?", "8", ("3", "6", "9"),
          "Each element is in or out: 2³ = 8, counting the empty set and the set itself."),
        Q("A relation that is reflexive, symmetric and transitive is",
          "an equivalence relation", ("a function", "a partial order", "always empty"),
          "Those three properties together define an equivalence relation."),
        Q("If f(x) = 2x + 3, then f⁻¹(x) is", "(x − 3)/2",
          ("(x + 3)/2", "2x − 3", "1/(2x + 3)"),
          "Solve y = 2x + 3 for x: x = (y − 3)/2."),
    ),
    "complex_quadratic": (
        Q("What is i²?", "−1", ("1", "i", "−i"), "i is defined as √−1."),
        Q("What are the roots of x² − 5x + 6 = 0?", "2 and 3",
          ("−2 and −3", "1 and 6", "−1 and 6"), "x² − 5x + 6 = (x − 2)(x − 3)."),
        Q("A quadratic ax² + bx + c = 0 with real coefficients has equal roots when",
          "b² − 4ac = 0", ("b² − 4ac > 0", "b² − 4ac < 0", "a = 0"),
          "A zero discriminant gives one repeated root."),
    ),
    "matrices": (
        Q("What is the determinant of the matrix [[2, 3], [1, 4]]?", "5",
          ("8", "11", "−5"), "2 × 4 − 3 × 1 = 5."),
        Q("A square matrix A has an inverse only if", "det A ≠ 0",
          ("det A = 0", "A is symmetric", "A has no zero entries"),
          "A⁻¹ = adj A / det A, which needs a non-zero determinant."),
        Q("A is a 2 × 3 matrix and B is 3 × 4. What is the order of AB?", "2 × 4",
          ("3 × 3", "4 × 2", "AB is not defined"),
          "Inner orders match (3 and 3); the outer ones give the result."),
    ),
    "perm_comb": (
        Q("In how many ways can 4 different books be arranged on a shelf?", "24",
          ("4", "16", "12"), "4! = 4 × 3 × 2 × 1 = 24."),
        Q("What is the value of ⁵C₂?", "10", ("20", "5", "25"),
          "5! / (2! × 3!) = 10."),
        Q("In how many ways can a committee of 2 be chosen from 6 people?", "15",
          ("12", "30", "36"), "Order does not matter: ⁶C₂ = 15."),
    ),
    "binomial": (
        Q("How many terms are there in the expansion of (a + b)⁸?", "9",
          ("8", "7", "16"), "An expansion of power n has n + 1 terms."),
        Q("What is the coefficient of x² in (1 + x)⁵?", "10", ("5", "20", "25"),
          "⁵C₂ = 10."),
        Q("The sum of the binomial coefficients in (1 + x)ⁿ is", "2ⁿ",
          ("n", "2n", "n²"), "Put x = 1."),
    ),
    "sequences": (
        Q("What is the 10th term of the arithmetic progression 2, 5, 8, ...?", "29",
          ("27", "30", "32"), "a + 9d = 2 + 9 × 3 = 29."),
        Q("What is the sum of the infinite series 1 + 1/2 + 1/4 + ...?", "2",
          ("1", "3", "It has no finite sum"), "a/(1 − r) = 1/(1 − 1/2) = 2."),
        Q("What is the geometric mean of 4 and 16?", "8", ("10", "12", "20"),
          "√(4 × 16) = 8. The arithmetic mean would be 10."),
    ),
    "limits": (
        Q("What is the limit of (sin x)/x as x tends to 0?", "1",
          ("0", "Infinity", "It does not exist"), "A standard limit, with x in radians."),
        Q("What is the derivative of x³?", "3x²", ("x²", "3x³", "x⁴/4"),
          "d/dx of xⁿ is nxⁿ⁻¹."),
        Q("At x = 0, the function f(x) = |x| is", "continuous but not differentiable",
          ("differentiable but not continuous", "neither continuous nor differentiable",
           "both continuous and differentiable"),
          "Its graph has a corner at 0: the slopes from the left and right differ."),
    ),
    "integration": (
        Q("∫ 2x dx is", "x² + C", ("2x² + C", "x + C", "2 + C"),
          "Differentiate x² to check: you get 2x."),
        Q("What is the value of ∫ x dx from 0 to 1?", "1/2", ("0", "1", "2"),
          "[x²/2] from 0 to 1 = 1/2."),
        Q("∫ eˣ dx is", "eˣ + C", ("xeˣ + C", "eˣ/x + C", "ln x + C"),
          "eˣ is its own derivative."),
    ),
    "diff_eq": (
        Q("What is the order of d²y/dx² + 3 dy/dx + y = 0?", "2", ("1", "3", "0"),
          "The order is that of the highest derivative."),
        Q("The general solution of dy/dx = 2x is", "y = x² + C",
          ("y = 2x + C", "y = x + C", "y = 2 + C"), "Integrate both sides."),
        Q("Which function solves dy/dx = y with y(0) = 1?", "y = eˣ",
          ("y = x", "y = e⁻ˣ", "y = x + 1"),
          "eˣ is its own derivative, and e⁰ = 1."),
    ),
    "coord_geom": (
        Q("What is the slope of the line through (1, 2) and (3, 6)?", "2",
          ("1", "3", "4"), "(6 − 2)/(3 − 1) = 2."),
        Q("What is the distance between (0, 0) and (3, 4)?", "5",
          ("7", "12", "25"), "√(3² + 4²) = 5."),
        Q("What is the centre of the circle x² + y² − 4x + 6y − 3 = 0?", "(2, −3)",
          ("(−2, 3)", "(4, −6)", "(−4, 6)"),
          "For x² + y² + 2gx + 2fy + c = 0 the centre is (−g, −f): here g = −2, f = 3."),
    ),
    "three_d": (
        Q("How far is the point (1, 2, 2) from the origin?", "3",
          ("5", "9", "√5"), "√(1 + 4 + 4) = 3."),
        Q("The direction cosines l, m, n of any line satisfy", "l² + m² + n² = 1",
          ("l + m + n = 1", "lmn = 1", "l = m = n"),
          "They are the components of a unit vector along the line."),
        Q("The plane x = 0 is the", "yz-plane", ("xy-plane", "zx-plane", "x-axis"),
          "Every point on it has x = 0 and any y and z."),
    ),
    "vectors": (
        Q("Two non-zero vectors have a dot product of 0. They are", "perpendicular",
          ("parallel", "equal", "opposite"), "a · b = |a||b| cos θ, and cos 90° = 0."),
        Q("What is the magnitude of 3i + 4j?", "5", ("7", "12", "25"),
          "√(3² + 4²) = 5."),
        Q("What is i × j?", "k", ("−k", "0", "1"),
          "The unit vectors follow the cycle i, j, k."),
    ),
    "stats_prob": (
        Q("A fair die is thrown once. What is the probability of an even number?", "1/2",
          ("1/6", "1/3", "2/3"), "Three of the six faces are even."),
        Q("What is the mean of 2, 4, 6, 8 and 10?", "6", ("5", "7", "8"),
          "30 / 5 = 6."),
        Q("If A and B are independent events, P(A and B) equals", "P(A) × P(B)",
          ("P(A) + P(B)", "P(A) / P(B)", "0"),
          "That is the definition of independence."),
    ),
    "trigonometry": (
        Q("sin²θ + cos²θ equals", "1", ("0", "2", "tan θ"),
          "Pythagoras on the unit circle."),
        Q("What is sin 30°?", "1/2", ("√3/2", "1", "1/√2"), "A standard value."),
        Q("What is tan 45°?", "1", ("0", "√3", "1/√3"),
          "sin 45° and cos 45° are equal."),
    ),
    # ----------------------------------------------------------------- biology
    "diversity": (
        Q("Who proposed the five kingdom classification?", "R. H. Whittaker",
          ("Carl Linnaeus", "Charles Darwin", "Gregor Mendel"),
          "Whittaker, in 1969: Monera, Protista, Fungi, Plantae and Animalia."),
        Q("Binomial nomenclature names each organism by its",
          "genus and species", ("family and order", "kingdom and phylum",
                                "common and local names"),
          "For example Homo sapiens or Mangifera indica."),
        Q("In the five kingdom system, bacteria belong to", "Monera",
          ("Protista", "Fungi", "Plantae"), "Monera holds the prokaryotes."),
    ),
    "plant_structure": (
        Q("Which tissue carries water from the roots to the leaves?", "Xylem",
          ("Phloem", "Cambium", "Epidermis"), "Phloem carries food instead."),
        Q("Tap roots are typical of", "dicots", ("monocots", "ferns", "mosses"),
          "Monocots usually have fibrous roots."),
        Q("Which tissue produces secondary growth in a dicot stem?", "Vascular cambium",
          ("Apical meristem", "Intercalary meristem", "Epidermis"),
          "This lateral meristem adds new xylem and phloem each year."),
    ),
    "cell": (
        Q("Which organelle is the main site of aerobic respiration?", "Mitochondrion",
          ("Nucleus", "Ribosome", "Golgi body"),
          "It makes most of the cell's ATP."),
        Q("Ribosomes are the site of", "protein synthesis",
          ("photosynthesis", "lipid storage", "glycolysis"),
          "They read messenger RNA and join amino acids."),
        Q("In which stage of mitosis do chromosomes line up at the middle of the cell?",
          "Metaphase", ("Prophase", "Anaphase", "Telophase"),
          "They gather on the metaphase plate before separating."),
    ),
    "plant_physiology": (
        Q("The light reactions of photosynthesis take place in the",
          "thylakoid membranes", ("stroma", "mitochondria", "cytoplasm"),
          "The Calvin cycle runs in the stroma."),
        Q("Which plant hormone promotes cell elongation and apical dominance?", "Auxin",
          ("Abscisic acid", "Ethylene", "Cytokinin"),
          "Auxin from the shoot tip suppresses side buds."),
        Q("The first stable product of the Calvin cycle in C3 plants is",
          "3-phosphoglyceric acid", ("oxaloacetic acid", "glucose", "pyruvic acid"),
          "Its three carbons give C3 plants their name. Oxaloacetic acid is the C4 product."),
    ),
    "plant_reproduction": (
        Q("Double fertilisation is a feature of", "angiosperms",
          ("gymnosperms", "bryophytes", "pteridophytes"),
          "One sperm fuses with the egg and another with the polar nuclei."),
        Q("The endosperm of a flowering plant is usually", "triploid",
          ("haploid", "diploid", "tetraploid"),
          "One male nucleus plus two polar nuclei: 3n."),
        Q("Pollen moving from the anther to the stigma of the same flower is called",
          "autogamy", ("xenogamy", "geitonogamy", "apomixis"),
          "Geitonogamy is another flower on the same plant; xenogamy is another plant."),
    ),
    "ecology": (
        Q("In a food chain, green plants are the", "producers",
          ("primary consumers", "decomposers", "secondary consumers"),
          "They make food from sunlight."),
        Q("About how much energy passes from one trophic level to the next?", "10%",
          ("1%", "50%", "90%"), "Lindeman's ten percent law."),
        Q("An interaction in which both species benefit is", "mutualism",
          ("parasitism", "commensalism", "competition"),
          "In commensalism only one benefits and the other is unaffected."),
    ),
    "animal_structure": (
        Q("Which tissue lines the inside of blood vessels?", "Squamous epithelium",
          ("Cartilage", "Skeletal muscle", "Adipose tissue"),
          "A single flat layer, the endothelium."),
        Q("Cardiac muscle is", "involuntary and striated",
          ("voluntary and striated", "involuntary and unstriated",
           "voluntary and unstriated"),
          "Striated like skeletal muscle, but not under conscious control."),
        Q("Bone and cartilage are", "connective tissue",
          ("epithelial tissue", "muscle tissue", "nervous tissue"),
          "Specialised connective tissues that support the body."),
    ),
    "human_physiology": (
        Q("Which blood cells carry oxygen?", "Red blood cells",
          ("White blood cells", "Platelets", "Plasma cells"),
          "Their haemoglobin binds oxygen."),
        Q("The working unit of the kidney is the", "nephron",
          ("neuron", "alveolus", "villus"), "Each kidney has about a million."),
        Q("Insulin is made by the", "pancreas", ("thyroid", "adrenal gland", "pituitary"),
          "By the beta cells of the islets of Langerhans."),
    ),
    "human_reproduction": (
        Q("Where does fertilisation normally happen in humans?", "In the fallopian tube",
          ("In the uterus", "In the ovary", "In the cervix"),
          "In the ampulla of the fallopian tube."),
        Q("Sperm are made in the", "seminiferous tubules",
          ("prostate gland", "vas deferens", "urethra"),
          "Inside the testes."),
        Q("A pregnancy test looks for which hormone?", "hCG",
          ("Insulin", "Thyroxine", "Adrenaline"),
          "Human chorionic gonadotropin, made after implantation."),
    ),
    "genetics": (
        Q("In Mendel's monohybrid cross, the F2 generation shows a phenotype ratio of",
          "3:1", ("1:1", "9:3:3:1", "1:2:1"),
          "1:2:1 is the genotype ratio; 9:3:3:1 is the dihybrid one."),
        Q("DNA replication is", "semi-conservative",
          ("conservative", "dispersive", "random"),
          "Each new molecule keeps one old strand, as Meselson and Stahl showed."),
        Q("Natural selection as the mechanism of evolution was proposed by",
          "Charles Darwin", ("Jean-Baptiste Lamarck", "Gregor Mendel", "Hugo de Vries"),
          "Lamarck proposed inheritance of acquired characters."),
    ),
    "human_welfare": (
        Q("Malaria is caused by", "Plasmodium, a protozoan",
          ("a virus", "a bacterium", "a fungus"),
          "Spread by the female Anopheles mosquito."),
        Q("Which cells does HIV mainly attack?", "Helper T lymphocytes",
          ("Red blood cells", "Platelets", "Liver cells"),
          "Losing them weakens the whole immune response."),
        Q("Penicillin was discovered by", "Alexander Fleming",
          ("Louis Pasteur", "Robert Koch", "Edward Jenner"),
          "In 1928, from the mould Penicillium."),
    ),
    "biotech": (
        Q("Restriction enzymes are used in genetic engineering to",
          "cut DNA at particular sequences",
          ("join DNA fragments", "copy RNA into DNA", "separate proteins"),
          "They are molecular scissors; ligase is the glue."),
        Q("Which enzyme joins DNA fragments together?", "DNA ligase",
          ("Restriction endonuclease", "Amylase", "Helicase"),
          "It seals the sugar-phosphate backbone."),
        Q("PCR is used to", "make many copies of a piece of DNA",
          ("cut DNA", "work out a protein's sequence", "stain chromosomes"),
          "Polymerase chain reaction doubles the target each cycle."),
    ),
}


# (section, unit) as written in elpr/exams.py -> the topics that check it.
UNIT_TOPICS: dict[tuple[str, str], tuple[str, ...]] = {
    # JEE Main and NEET UG physics, which name some units differently
    ("Physics", "Units and Measurements"): ("measurement",),
    ("Physics", "Physics and Measurement"): ("measurement",),
    ("Physics", "Kinematics"): ("kinematics",),
    ("Physics", "Laws of Motion"): ("laws_of_motion",),
    ("Physics", "Work, Energy and Power"): ("work_energy",),
    ("Physics", "Rotational Motion"): ("rotation",),
    ("Physics", "Gravitation"): ("gravitation",),
    ("Physics", "Properties of Solids and Liquids"): ("bulk_matter",),
    ("Physics", "Thermodynamics"): ("phys_thermo",),
    ("Physics", "Kinetic Theory of Gases"): ("kinetic_theory",),
    ("Physics", "Oscillations and Waves"): ("oscillations_waves",),
    ("Physics", "Electrostatics"): ("electrostatics",),
    ("Physics", "Current Electricity"): ("current_electricity",),
    ("Physics", "Magnetic Effects of Current and Magnetism"): ("magnetism",),
    ("Physics", "Electromagnetic Induction and Alternating Currents"): ("emi_ac",),
    ("Physics", "Electromagnetic Waves"): ("em_waves",),
    ("Physics", "Optics"): ("optics",),
    ("Physics", "Dual Nature of Matter and Radiation"): ("dual_nature",),
    ("Physics", "Atoms and Nuclei"): ("atoms_nuclei",),
    ("Physics", "Electronic Devices"): ("electronic_devices",),
    ("Physics", "Experimental Skills"): ("experimental",),
    # chemistry
    ("Chemistry", "Some Basic Concepts in Chemistry"): ("basic_concepts",),
    ("Chemistry", "Atomic Structure"): ("atomic_structure",),
    ("Chemistry", "Chemical Bonding and Molecular Structure"): ("bonding",),
    ("Chemistry", "Chemical Thermodynamics"): ("chem_thermo",),
    ("Chemistry", "Solutions"): ("solutions",),
    ("Chemistry", "Equilibrium"): ("equilibrium",),
    ("Chemistry", "Redox Reactions and Electrochemistry"): ("redox", "electrochem"),
    ("Chemistry", "Chemical Kinetics"): ("kinetics",),
    ("Chemistry", "Classification of Elements and Periodicity in Properties"):
        ("periodicity",),
    ("Chemistry", "p-Block Elements"): ("p_block",),
    ("Chemistry", "d- and f-Block Elements"): ("d_f_block",),
    ("Chemistry", "Co-ordination Compounds"): ("coordination",),
    ("Chemistry", "Coordination Compounds"): ("coordination",),
    ("Chemistry", "Purification and Characterisation of Organic Compounds"):
        ("purification",),
    ("Chemistry", "Some Basic Principles of Organic Chemistry"): ("goc",),
    ("Chemistry", "Hydrocarbons"): ("hydrocarbons",),
    ("Chemistry", "Organic Compounds Containing Halogens"): ("halogens",),
    ("Chemistry", "Organic Compounds Containing Oxygen"): ("alcohols", "carbonyls"),
    ("Chemistry", "Organic Compounds Containing Nitrogen"): ("amines",),
    ("Chemistry", "Biomolecules"): ("biomolecules",),
    ("Chemistry", "Principles Related to Practical Chemistry"): ("practical",),
    # JEE Main mathematics
    ("Mathematics", "Sets, Relations and Functions"): ("sets",),
    ("Mathematics", "Complex Numbers and Quadratic Equations"): ("complex_quadratic",),
    ("Mathematics", "Matrices and Determinants"): ("matrices",),
    ("Mathematics", "Permutations and Combinations"): ("perm_comb",),
    ("Mathematics", "Binomial Theorem and its Simple Applications"): ("binomial",),
    ("Mathematics", "Sequence and Series"): ("sequences",),
    ("Mathematics", "Limit, Continuity and Differentiability"): ("limits",),
    ("Mathematics", "Integral Calculus"): ("integration",),
    ("Mathematics", "Differential Equations"): ("diff_eq",),
    ("Mathematics", "Co-ordinate Geometry"): ("coord_geom",),
    ("Mathematics", "Three Dimensional Geometry"): ("three_d",),
    ("Mathematics", "Vector Algebra"): ("vectors",),
    ("Mathematics", "Statistics and Probability"): ("stats_prob",),
    ("Mathematics", "Trigonometry"): ("trigonometry",),
    # NEET UG biology, the ten units of the NMC syllabus. Two of them span both halves
    # of the paper, so they carry both sets of questions.
    ("Biology", "Diversity in Living World"): ("diversity",),
    ("Biology", "Structural Organisation in Animals and Plants"): ("plant_structure", "animal_structure"),
    ("Biology", "Cell Structure and Function"): ("cell",),
    ("Biology", "Plant Physiology"): ("plant_physiology",),
    ("Biology", "Human Physiology"): ("human_physiology",),
    ("Biology", "Reproduction"): ("plant_reproduction", "human_reproduction"),
    ("Biology", "Genetics and Evolution"): ("genetics",),
    ("Biology", "Biology and Human Welfare"): ("human_welfare",),
    ("Biology", "Biotechnology and Its Applications"): ("biotech",),
    ("Biology", "Ecology and Environment"): ("ecology",),
}

COVERAGE_NOTE = ("Self-check questions cover every JEE Main and NEET UG unit. CUET's "
                 "language, General Aptitude Test and humanities and commerce papers have "
                 "none yet.")


def has_check(section: str, unit: str) -> bool:
    return (section, unit) in UNIT_TOPICS


def _questions(section: str, unit: str) -> list[tuple[str, Question]]:
    topics = UNIT_TOPICS.get((section, unit))
    if topics is None:
        raise KeyError((section, unit))
    return [(f"{t}:{i}", q) for t in topics for i, q in enumerate(BANK[t])]


def _order(qid: str) -> list[int]:
    """Where each stored option goes on the page: the same shuffle every time.

    Index 0 is the right answer in storage. crc32 rather than hash(), because Python
    salts string hashes per process and the order must survive a restart."""
    order = [0, 1, 2, 3]
    random.Random(zlib.crc32(qid.encode())).shuffle(order)
    return order


def _shown(q: Question, qid: str) -> list[str]:
    stored = [q.right, *q.wrong]
    return [stored[i] for i in _order(qid)]


def _check_unit(exam_key: str, section: str, unit: str) -> None:
    """The unit must belong to this exam, so a check is always about something on the
    syllabus the student picked."""
    sections = dict(exams.sections_for(exam_key))        # raises KeyError for bad exams
    if unit not in sections.get(section, ()):
        raise KeyError((exam_key, section, unit))


def questions_for(exam_key: str, section: str, unit: str) -> dict:
    """The questions for one unit, without their answers."""
    _check_unit(exam_key, section, unit)
    items = _questions(section, unit)
    return {
        "exam": exam_key, "section": section, "unit": unit, "n": len(items),
        "pass_mark": PASS_MARK,
        "questions": [{"id": qid, "prompt": q.prompt, "options": _shown(q, qid)}
                      for qid, q in items],
        "note": "Short questions to see whether a unit has stuck. Not a mock paper, and "
                "not a prediction of an exam score.",
    }


def grade(exam_key: str, section: str, unit: str, answers: dict) -> dict:
    """Mark a set of answers. `answers` maps question id to the index of the option
    the student chose, as the options were shown. Every question must be answered."""
    _check_unit(exam_key, section, unit)
    items = _questions(section, unit)
    missing = [qid for qid, _ in items if qid not in answers]
    if missing:
        raise ValueError("answer every question first")
    results, right = [], 0
    for qid, q in items:
        order = _order(qid)
        try:
            chosen = int(answers[qid])
        except (TypeError, ValueError) as exc:
            raise ValueError("each answer is the number of an option") from exc
        if not 0 <= chosen < 4:
            raise ValueError("each answer is the number of an option")
        correct_at = order.index(0)
        ok = chosen == correct_at
        right += ok
        results.append({"id": qid, "chosen": chosen, "answer": correct_at,
                        "answer_text": q.right, "correct": ok, "why": q.why})
    score = round(100 * right / len(items))
    return {"exam": exam_key, "section": section, "unit": unit, "results": results,
            "n": len(items), "n_correct": right, "score": score,
            "passed": score >= PASS_MARK}
