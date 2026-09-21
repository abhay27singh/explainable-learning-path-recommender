"""Course Finder: suggests courses to a student from their level, background and interests.

This is RULE-BASED MATCHING, not the machine-learning model. The knowledge tracing model
was trained on UK Open University data and knows nothing about Indian students or
courses, and no dataset of Indian students' interests and outcomes was available to learn
from. Every suggestion comes from the rules below, which anyone can read and check.

Levels:
    diploma_10  Diploma after class 10 (for example polytechnic engineering diplomas)
    diploma_12  Diploma after class 12
    ug          Undergraduate degree after class 12
    pg          Postgraduate study after a bachelor's degree

For diplomas after class 12 and undergraduate degrees, eligibility is checked against the
class 12 subjects the student actually took. For postgraduate study it is checked against
the student's bachelor's degree. Eligibility, entrance exams and subjects describe common
national patterns (AICTE, UGC, NMC, PCI, BCI, NCTE, COA, NCHMCT) and vary by university
and board; the interface says so. Subjects are grouped by year, not by week, because
weekly schedules are set by each institution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from urllib.parse import quote_plus

METHOD = "Rule-based matching, not the machine-learning model"
NOTE = ("Eligibility, entrance exams and subjects vary by university and board. Always "
        "check the institution's own admission rules and syllabus.")
SUBJECTS_NOTE = "Common CBSE combinations. State boards and schools may differ."

LEVELS: dict[str, str] = {
    "skill": "Skill course",
    "diploma_10": "Diploma after class 10",
    "diploma_12": "Diploma after class 12",
    "ug": "Undergraduate degree",
    "pg": "Postgraduate degree",
}
# what the student is asked for at each level
LEVEL_INPUT: dict[str, str] = {
    "skill": "none",
    "diploma_10": "none", "diploma_12": "stream", "ug": "stream", "pg": "degree",
}

# NEP 2020 removes the hard separation between academic and vocational study, so skill
# courses sit beside the academic ladder rather than below it: they can be taken from
# any rung, alongside school or a degree.
TRACKS: dict[str, str] = {
    "academic": "Academic path",
    "skill": "Skill path",
}
SKILL_NOTE = ("Skill courses under the National Skills Qualifications Framework (NSQF). "
              "NEP 2020 removes the separation between academic and vocational study, so "
              "these can be taken alongside school or a degree, not instead of one.")
# What a degree is worth if a student leaves part way, under NEP 2020's multiple entry
# and exit rules. Institutions apply these through the Academic Bank of Credits.
NEP_UG_EXITS: tuple[str, ...] = (
    "Leave after 1 year: certificate",
    "Leave after 2 years: diploma",
    "Leave after 3 years: bachelor's degree",
    "Finish 4 years: bachelor's degree with research",
)

# Where a student is now. Students at a course stage are enrolled on a course and get a
# week-by-week path; school students get Course Finder suggestions for the next level.
STAGES: dict[str, str] = {
    "class_10": "Class 10", "class_12": "Class 12", "diploma": "Diploma",
    "ug": "Undergraduate", "pg": "Postgraduate",
}
COURSE_STAGES: tuple[str, ...] = ("diploma", "ug", "pg")
# the levels the Course Finder offers from each stage
NEXT_LEVELS: dict[str, tuple[str, ...]] = {
    "class_10": ("diploma_10", "skill"), "class_12": ("diploma_12", "ug", "skill"),
    "diploma": ("ug", "skill"), "ug": ("pg", "skill"), "pg": ("skill",),
}

# What a class 10 student is studying now. Same idea as STREAM_SUBJECTS one rung up.
CLASS10_SUBJECTS: dict[str, list[str]] = {
    "core": ["English", "Mathematics", "Science", "Social Science"],
    "optional": ["Hindi", "Sanskrit", "Computer Applications", "Information Technology"],
}

STREAMS: dict[str, str] = {
    "pcm": "Science with Mathematics (PCM)",
    "pcb": "Science with Biology (PCB)",
    "pcmb": "Science with Mathematics and Biology (PCMB)",
    "commerce": "Commerce",
    "arts": "Arts / Humanities",
}

STREAM_SUBJECTS: dict[str, dict[str, list[str]]] = {
    "pcm": {"core": ["English", "Physics", "Chemistry", "Mathematics"],
            "optional": ["Computer Science", "Informatics Practices", "Physical Education",
                         "Economics"]},
    "pcb": {"core": ["English", "Physics", "Chemistry", "Biology"],
            "optional": ["Mathematics", "Psychology", "Physical Education",
                         "Computer Science"]},
    "pcmb": {"core": ["English", "Physics", "Chemistry", "Mathematics", "Biology"],
             "optional": ["Computer Science", "Physical Education"]},
    "commerce": {"core": ["English", "Accountancy", "Business Studies", "Economics"],
                 "optional": ["Mathematics", "Informatics Practices", "Entrepreneurship",
                              "Physical Education"]},
    "arts": {"core": ["English", "History", "Political Science"],
             "optional": ["Geography", "Economics", "Psychology", "Sociology",
                          "Mathematics", "Fine Arts"]},
}

DEGREES: dict[str, str] = {
    "btech_cs": "B.Tech / B.E. in Computer Science or IT",
    "btech_other": "B.Tech / B.E. in another branch",
    "bca": "BCA",
    "bsc_cs": "B.Sc Computer Science",
    "bsc_maths": "B.Sc Mathematics, Physics or Statistics",
    "bsc_life": "B.Sc Life Sciences or Biotechnology",
    "bpharm": "B.Pharm",
    "bcom": "B.Com",
    "bba": "BBA",
    "ba_econ": "B.A. Economics",
    "ba_psych": "B.A. Psychology",
    "ba_english": "B.A. English",
    "ba_other": "B.A. in another subject",
}

INTERESTS: dict[str, str] = {
    "coding": "Coding and computers",
    "maths": "Mathematics and problem solving",
    "electronics": "Electronics and circuits",
    "machines": "Machines and how things work",
    "biology": "Living things and biology",
    "health": "Health and medicine",
    "business": "Business and starting companies",
    "finance": "Money, accounts and finance",
    "people": "Understanding people and behaviour",
    "society": "Society, economy and policy",
    "data": "Data and statistics",
    "design": "Design and creativity",
    "law": "Law and justice",
    "teaching": "Teaching and education",
    "media": "Media, writing and communication",
    "hospitality": "Hospitality, travel and food",
}

# subjects that are placements or project work rather than something to look up
_NOT_A_TOPIC = re.compile(
    r"internship|project|dissertation|portfolio|thesis|practicum|training|moot|studio|"
    r"electives?\b|specialisation", re.I)


def _search(query: str) -> str:
    return "https://www.google.com/search?q=" + quote_plus(query)


def _youtube(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + quote_plus(query)


def study_links(subject: str, schooling: str | None = None,
                track: str = "academic") -> dict[str, str]:
    """Free study links for one subject, or {} for placements and projects.

    Which sites depends on who is studying. NPTEL and SWAYAM are university platforms
    and carry nothing for class 10 or class 12, so school subjects link to NCERT, Khan
    Academy and YouTube instead. YouTube is offered at every level.

    NPTEL, SWAYAM and NCERT do not honour a search term in their own URLs (checked in a
    browser), so those are Google searches restricted to each site. YouTube does, so
    that one is a real YouTube search."""
    if _NOT_A_TOPIC.search(subject):
        return {}
    phrase = f'"{subject}"'
    if track == "skill":               # trade skills: Skill India, and demonstrations
        return {
            "skillindia": _search(f"site:skillindia.gov.in {phrase}"),
            "nsdc": _search(f"site:nsdcindia.org OR site:skillindiadigital.gov.in {phrase}"),
            "youtube": _youtube(f"{subject} training practical demonstration"),
        }
    if schooling:                      # "class 10" or "class 12"
        return {
            "ncert": _search(f"site:ncert.nic.in {phrase} {schooling}"),
            "khan": _search(f"site:khanacademy.org {phrase}"),
            "youtube": _youtube(f"{subject} {schooling} full chapter explanation"),
        }
    return {
        "nptel": _search(f"site:nptel.ac.in {phrase}"),
        "swayam": _search(f"site:swayam.gov.in {phrase}"),
        "youtube": _youtube(f"{subject} full course lecture"),
    }


@dataclass(frozen=True)
class Course:
    key: str
    name: str
    level: str
    category: str
    duration: str
    framework: str
    eligibility: str
    interests: dict           # interest key -> weight 1 to 3
    years: tuple              # ((label, (subjects...)), ...)
    needs_all: tuple = ()     # class 12 subjects that are all required
    needs_any: tuple = ()     # at least one of these class 12 subjects is required
    degrees: frozenset | None = None   # postgraduate: qualifying degrees (None = any degree)

    def eligible(self, subjects: set[str], degree: str | None) -> bool:
        if self.level in ("diploma_10", "skill"):
            return True
        if self.level == "pg":
            return self.degrees is None or degree in self.degrees
        if not set(self.needs_all) <= subjects:
            return False
        return not self.needs_any or bool(set(self.needs_any) & subjects)

    @property
    def track(self) -> str:
        return "skill" if self.level == "skill" else "academic"

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "level": self.level,
            "level_label": LEVELS[self.level], "category": self.category,
            "duration": self.duration, "framework": self.framework,
            "eligibility": self.eligibility, "track": self.track,
            "track_label": TRACKS[self.track],
            # NEP 2020 lets a degree student leave with a qualification at each year
            "exits": list(NEP_UG_EXITS) if self.level == "ug" else [],
            "years": [{"year": label,
                       "subjects": [{"name": s, "links": study_links(s, track=self.track)}
                                    for s in subjects]}
                      for label, subjects in self.years],
        }


ENGINEERING_YEAR_1 = ("Engineering Mathematics", "Engineering Physics", "Engineering Chemistry",
                      "Programming for Problem Solving", "Basic Electrical Engineering",
                      "Engineering Graphics and Design")
POLYTECHNIC_YEAR_1 = ("Applied Mathematics", "Applied Physics", "Applied Chemistry",
                      "Communication Skills", "Engineering Drawing")
JEE = "Physics and Mathematics in class 12, usually with an entrance exam such as JEE Main"
THREE_OR_FOUR = "3 years (4 with honours under NEP 2020)"
LATERAL = ("Class 10 pass, usually with Mathematics and Science. Lateral entry to the second "
           "year of a B.Tech is often possible afterwards")

ITI = "ITI trade certificate under the NSQF, awarded by NCVT or SCVT"
NSDC = "NSQF qualification pack from the relevant Sector Skill Council"
SKILL_ANY = "Class 8 or class 10 pass, depending on the trade and the centre"

COURSES: tuple[Course, ...] = (
    # ---- skill path: NSQF trades and job roles, open from any rung of the ladder ----
    Course("iti_electrician", "ITI Electrician", "skill", "Trades and technology",
           "2 years", ITI, "Class 10 pass with Science and Mathematics",
           {"electronics": 3, "machines": 2},
           (("Year 1: Basics of the trade",
             ("Electrical Safety and Tools", "Basic Electrical Practice", "Wiring and Circuits",
              "Measuring Instruments", "Workshop Calculation and Science")),
            ("Year 2: On the job",
             ("Motors and Transformers", "Domestic and Industrial Wiring",
              "Motor Winding and Repair", "Panel and Control Wiring", "Employability Skills")))),
    Course("iti_fitter", "ITI Fitter", "skill", "Trades and technology",
           "2 years", ITI, "Class 10 pass with Science and Mathematics",
           {"machines": 3, "design": 1},
           (("Year 1: Bench and machine work",
             ("Fitting Hand Tools and Safety", "Measurement and Marking", "Drilling and Grinding",
              "Engineering Drawing", "Workshop Calculation and Science")),
            ("Year 2: Assembly and maintenance",
             ("Machine Assembly and Alignment", "Pipe Fitting", "Bearings and Lubrication",
              "Preventive Maintenance", "Employability Skills")))),
    Course("iti_welder", "ITI Welder", "skill", "Trades and technology",
           "1 year", ITI, "Class 8 or class 10 pass, depending on the state",
           {"machines": 3},
           (("Core trade practice",
             ("Welding Safety and Equipment", "Arc Welding", "Gas Welding and Cutting",
              "MIG and TIG Welding", "Welding Defects and Inspection", "Employability Skills")),)),
    Course("iti_copa", "ITI Computer Operator and Programming Assistant (COPA)", "skill",
           "Digital and office skills", "1 year", ITI, "Class 10 pass",
           {"coding": 2, "business": 1, "data": 1},
           (("Core trade practice",
             ("Computer Fundamentals and Operating Systems", "Office Productivity Software",
              "Typing and Data Entry", "Web Design Basics", "Database Basics",
              "Accounting with Tally", "Employability Skills")),)),
    Course("skill_solar", "Solar Panel Installation Technician", "skill",
           "Green and energy skills", "3 to 6 months", NSDC, SKILL_ANY,
           {"electronics": 3, "machines": 2},
           (("Qualification pack",
             ("Solar Energy Basics", "Panel Mounting and Site Survey", "Wiring and Inverters",
              "Testing and Commissioning", "Safety at Height", "Maintenance and Fault Finding")),)),
    Course("skill_healthcare", "General Duty Assistant (healthcare)", "skill",
           "Healthcare skills", "6 to 12 months", NSDC, "Class 10 pass",
           {"health": 3, "people": 2, "biology": 1},
           (("Qualification pack",
             ("Human Body Basics", "Patient Care and Hygiene", "Vital Signs and Monitoring",
              "Infection Control", "Medical Records and Communication", "First Aid and Emergency")),)),
    Course("skill_beauty", "Beauty Therapist", "skill", "Personal care and wellness",
           "3 to 6 months", NSDC, SKILL_ANY,
           {"design": 2, "people": 2, "health": 1},
           (("Qualification pack",
             ("Skin and Hair Science", "Salon Hygiene and Safety", "Facial and Skin Treatments",
              "Hair Care and Styling", "Client Consultation", "Salon Business Basics")),)),
    Course("skill_tailoring", "Self-Employed Tailor", "skill", "Craft and design skills",
           "3 to 6 months", NSDC, SKILL_ANY,
           {"design": 3, "business": 1},
           (("Qualification pack",
             ("Measurement and Body Shapes", "Pattern Making", "Cutting and Stitching",
              "Machine Care", "Finishing and Alterations", "Costing and Customer Orders")),)),
    Course("skill_agri", "Agriculture Extension Service Provider", "skill",
           "Agriculture skills", "3 to 6 months", NSDC, "Class 10 pass",
           {"biology": 3, "business": 1, "people": 1},
           (("Qualification pack",
             ("Soil and Crop Basics", "Seeds and Sowing", "Irrigation Methods",
              "Pest and Disease Management", "Farm Machinery", "Marketing Farm Produce")),)),
    Course("skill_retail", "Retail Sales Associate", "skill", "Business and service skills",
           "3 months", NSDC, "Class 10 pass",
           {"business": 3, "people": 2},
           (("Qualification pack",
             ("Retail Basics", "Product Knowledge", "Customer Service", "Billing and Point of Sale",
              "Stock and Display", "Workplace Communication")),)),
    Course("skill_data_entry", "Data Entry and Spreadsheet Operator", "skill",
           "Digital and office skills", "3 months", NSDC, "Class 10 pass",
           {"data": 2, "coding": 1, "business": 1},
           (("Qualification pack",
             ("Keyboard Skills and Accuracy", "Word Processing", "Spreadsheets and Formulas",
              "Charts and Reports", "Data Cleaning Basics", "Digital Safety")),)),
    Course("skill_digital_marketing", "Digital Marketing Executive", "skill",
           "Business and service skills", "3 to 6 months", NSDC, "Class 12 pass",
           {"media": 3, "business": 2, "data": 1},
           (("Qualification pack",
             ("Digital Marketing Basics", "Social Media Content", "Search Engine Optimisation",
              "Online Advertising", "Email and Messaging Campaigns", "Measuring Results")),)),
    Course("skill_web_dev", "Web Developer (front end)", "skill", "Digital and office skills",
           "6 months", NSDC, "Class 12 pass",
           {"coding": 3, "design": 2},
           (("Qualification pack",
             ("HTML and CSS", "JavaScript Basics", "Responsive Layouts", "Version Control with Git",
              "Working with APIs", "Accessibility and Testing")),)),
    Course("skill_data_analytics", "Data Analytics Associate", "skill",
           "Digital and office skills", "6 months", NSDC, "Class 12 pass, Mathematics helps",
           {"data": 3, "maths": 2, "coding": 1},
           (("Qualification pack",
             ("Spreadsheets for Analysis", "SQL and Databases", "Statistics for Analytics",
              "Python for Data", "Dashboards and Visualisation", "Reporting Findings")),)),
    Course("skill_hospitality", "Food and Beverage Service Steward", "skill",
           "Hospitality skills", "3 to 6 months", NSDC, "Class 10 pass",
           {"hospitality": 3, "people": 2},
           (("Qualification pack",
             ("Food Safety and Hygiene", "Table Service", "Menu and Beverage Knowledge",
              "Customer Handling", "Billing and Settlement", "Teamwork in Service")),)),
    Course("skill_teaching_aide", "Early Childhood Educator Assistant", "skill",
           "Teaching and care skills", "6 months", NSDC, "Class 12 pass",
           {"teaching": 3, "people": 2},
           (("Qualification pack",
             ("Child Development Basics", "Play-Based Learning", "Classroom Routines",
              "Storytelling and Language Activities", "Health and Safety of Children",
              "Working with Parents")),)),
    # ------------------------------------------------------------- diploma after class 10
    Course("dip_cs", "Diploma in Computer Engineering", "diploma_10", "Engineering and technology",
           "3 years", "AICTE-approved polytechnic curriculum", LATERAL,
           {"coding": 3, "electronics": 1},
           (("Year 1: Foundations", POLYTECHNIC_YEAR_1),
            ("Year 2: Core computing", ("Programming in C", "Data Structures",
             "Digital Electronics", "Database Management Systems")),
            ("Year 3: Applications", ("Computer Networks", "Operating Systems",
             "Web Technologies", "Industrial Training and Project")))),
    Course("dip_mech", "Diploma in Mechanical Engineering", "diploma_10",
           "Engineering and technology", "3 years", "AICTE-approved polytechnic curriculum",
           LATERAL, {"machines": 3, "design": 1},
           (("Year 1: Foundations", POLYTECHNIC_YEAR_1),
            ("Year 2: Core", ("Strength of Materials", "Thermal Engineering",
             "Manufacturing Processes", "Machine Drawing")),
            ("Year 3: Applications", ("Fluid Mechanics and Machinery", "Automobile Engineering",
             "Computer-Aided Drafting", "Industrial Training and Project")))),
    Course("dip_elec", "Diploma in Electrical Engineering", "diploma_10",
           "Engineering and technology", "3 years", "AICTE-approved polytechnic curriculum",
           LATERAL, {"electronics": 3, "machines": 1},
           (("Year 1: Foundations", POLYTECHNIC_YEAR_1),
            ("Year 2: Core", ("Electrical Circuits", "Electrical Machines",
             "Electrical Measurements", "Basic Electronics")),
            ("Year 3: Applications", ("Power Systems", "Electrical Installation and Estimation",
             "Industrial Training and Project")))),
    Course("dip_civil", "Diploma in Civil Engineering", "diploma_10",
           "Engineering and technology", "3 years", "AICTE-approved polytechnic curriculum",
           LATERAL, {"machines": 2, "design": 1},
           (("Year 1: Foundations", POLYTECHNIC_YEAR_1),
            ("Year 2: Core", ("Surveying", "Building Materials and Construction",
             "Mechanics of Structures", "Concrete Technology")),
            ("Year 3: Applications", ("Estimating and Costing", "Transportation Engineering",
             "Public Health Engineering", "Industrial Training and Project")))),

    # ------------------------------------------------------------- diploma after class 12
    Course("dpharm", "Diploma in Pharmacy (D.Pharm)", "diploma_12", "Health", "2 years",
           "Pharmacy Council of India regulations",
           "Physics and Chemistry with Mathematics or Biology in class 12",
           {"health": 3, "biology": 2},
           (("Year 1", ("Pharmaceutics", "Pharmaceutical Chemistry", "Pharmacognosy",
             "Human Anatomy and Physiology", "Social Pharmacy")),
            ("Year 2", ("Pharmacology", "Community Pharmacy and Management",
             "Biochemistry and Clinical Pathology", "Pharmacotherapeutics",
             "Hospital and Clinical Pharmacy", "Pharmacy Law and Ethics"))),
           needs_all=("Physics", "Chemistry"), needs_any=("Mathematics", "Biology")),
    Course("deled", "Diploma in Elementary Education (D.El.Ed)", "diploma_12", "Education",
           "2 years", "NCTE norms",
           "Class 12 pass in any stream, usually with a minimum percentage set by the state",
           {"teaching": 3, "people": 2},
           (("Year 1", ("Childhood and the Development of Children",
             "Understanding Language and Early Literacy",
             "Mathematics Education for the Primary School Child")),
            ("Year 2", ("Pedagogy of Environmental Studies",
             "Diversity, Gender and Inclusive Education", "School Internship")))),
    Course("dip_hotel", "Diploma in Hotel Management", "diploma_12", "Hospitality",
           "1 to 3 years, depending on the institute", "Institute syllabi",
           "Class 12 pass in any stream", {"hospitality": 3, "business": 1},
           (("Course content", ("Food Production", "Food and Beverage Service",
             "Front Office Operations", "Housekeeping", "Industrial Training")),)),
    Course("dca", "Diploma in Computer Applications (DCA)", "diploma_12", "Computing",
           "1 year", "Institute syllabi", "Class 12 pass in any stream",
           {"coding": 2, "data": 1, "business": 1},
           (("Course content", ("Computer Fundamentals", "Office Productivity Software",
             "Programming Basics", "Web Basics", "Database Basics")),)),
    Course("dip_graphic", "Diploma in Graphic Design", "diploma_12", "Design and architecture",
           "1 year", "Institute syllabi", "Class 12 pass in any stream",
           {"design": 3, "media": 2},
           (("Course content", ("Design Principles", "Typography", "Digital Illustration",
             "Layout and Branding", "Portfolio")),)),

    # ------------------------------------------------------------- undergraduate
    Course("btech_cse", "B.Tech Computer Science and Engineering", "ug",
           "Engineering and technology", "4 years", "AICTE model curriculum", JEE,
           {"coding": 3, "maths": 2, "data": 2, "electronics": 1},
           (("Year 1: Foundations", ENGINEERING_YEAR_1),
            ("Year 2: Core computing", ("Data Structures", "Discrete Mathematics",
             "Digital Logic Design", "Computer Organisation and Architecture",
             "Object-Oriented Programming")),
            ("Year 3: Systems", ("Operating Systems", "Database Management Systems",
             "Computer Networks", "Design and Analysis of Algorithms",
             "Theory of Computation", "Software Engineering")),
            ("Year 4: Specialisation", ("Compiler Design", "Machine Learning",
             "Internship", "Major Project"))),
           needs_all=("Physics", "Mathematics")),
    Course("btech_ece", "B.Tech Electronics and Communication Engineering", "ug",
           "Engineering and technology", "4 years", "AICTE model curriculum", JEE,
           {"electronics": 3, "maths": 2, "coding": 1, "machines": 1},
           (("Year 1: Foundations", ENGINEERING_YEAR_1),
            ("Year 2: Circuits and signals", ("Electronic Devices and Circuits",
             "Network Analysis", "Signals and Systems", "Digital Electronics")),
            ("Year 3: Communication and control", ("Analog Circuits",
             "Microprocessors and Microcontrollers", "Communication Systems",
             "Control Systems", "Electromagnetic Fields")),
            ("Year 4: Specialisation", ("VLSI Design", "Wireless Communication",
             "Embedded Systems", "Major Project"))),
           needs_all=("Physics", "Mathematics")),
    Course("btech_mech", "B.Tech Mechanical Engineering", "ug", "Engineering and technology",
           "4 years", "AICTE model curriculum", JEE,
           {"machines": 3, "maths": 2, "electronics": 1},
           (("Year 1: Foundations", ENGINEERING_YEAR_1),
            ("Year 2: Mechanics and materials", ("Engineering Mechanics", "Thermodynamics",
             "Strength of Materials", "Manufacturing Processes")),
            ("Year 3: Machines and energy", ("Fluid Mechanics", "Theory of Machines",
             "Heat Transfer", "Machine Design")),
            ("Year 4: Specialisation", ("CAD and CAM", "Industrial Engineering",
             "Major Project"))),
           needs_all=("Physics", "Mathematics")),
    Course("barch", "B.Arch (Architecture)", "ug", "Design and architecture", "5 years",
           "Council of Architecture norms",
           "Physics, Chemistry and Mathematics in class 12, with NATA or JEE Main Paper 2",
           {"design": 3, "machines": 1, "maths": 1},
           (("Year 1: Foundations", ("Architectural Design Studio", "Building Materials",
             "Architectural Drawing and Graphics", "History of Architecture")),
            ("Year 2: Construction", ("Building Construction", "Theory of Structures",
             "Climate-Responsive Design")),
            ("Year 3: Services", ("Building Services", "Working Drawings",
             "Landscape Design")),
            ("Year 4: Practice", ("Urban Design", "Professional Training")),
            ("Year 5: Thesis", ("Professional Practice", "Thesis Project"))),
           needs_all=("Physics", "Chemistry", "Mathematics")),
    Course("bca", "Bachelor of Computer Applications (BCA)", "ug", "Computing", THREE_OR_FOUR,
           "University syllabi",
           "Any stream at many universities; some require Mathematics in class 12",
           {"coding": 3, "data": 1, "business": 1},
           (("Year 1: Foundations", ("Programming in C", "Computer Fundamentals",
             "Mathematics for Computing", "Digital Electronics")),
            ("Year 2: Core computing", ("Data Structures", "Object-Oriented Programming",
             "Database Management Systems", "Operating Systems")),
            ("Year 3: Applications", ("Web Technologies", "Computer Networks",
             "Software Engineering", "Project")))),
    Course("bsc_cs", "B.Sc Computer Science", "ug", "Computing", THREE_OR_FOUR,
           "UGC undergraduate framework", "Mathematics in class 12",
           {"coding": 2, "maths": 2, "data": 2},
           (("Year 1: Foundations", ("Programming Fundamentals",
             "Computer System Architecture", "Discrete Structures", "Calculus")),
            ("Year 2: Core computing", ("Data Structures", "Database Management Systems",
             "Operating Systems", "Object-Oriented Programming")),
            ("Year 3: Advanced", ("Computer Networks", "Design and Analysis of Algorithms",
             "Software Engineering", "Project"))),
           needs_all=("Mathematics",)),
    Course("bsc_maths", "B.Sc Mathematics", "ug", "Sciences", THREE_OR_FOUR,
           "UGC undergraduate framework", "Mathematics in class 12",
           {"maths": 3, "data": 2},
           (("Year 1: Foundations", ("Calculus", "Algebra", "Analytic Geometry",
             "Differential Equations")),
            ("Year 2: Core", ("Real Analysis", "Linear Algebra", "Group Theory",
             "Numerical Methods")),
            ("Year 3: Advanced", ("Complex Analysis", "Probability and Statistics",
             "Ring Theory"))),
           needs_all=("Mathematics",)),
    Course("bsc_physics", "B.Sc Physics", "ug", "Sciences", THREE_OR_FOUR,
           "UGC undergraduate framework", "Physics and Mathematics in class 12",
           {"maths": 2, "machines": 2, "electronics": 1},
           (("Year 1: Foundations", ("Mechanics", "Mathematical Physics",
             "Electricity and Magnetism", "Waves and Optics")),
            ("Year 2: Core", ("Thermal Physics", "Analog and Digital Electronics",
             "Elements of Modern Physics")),
            ("Year 3: Advanced", ("Quantum Mechanics", "Solid State Physics",
             "Nuclear and Particle Physics", "Electromagnetic Theory"))),
           needs_all=("Physics", "Mathematics")),
    Course("bsc_biotech", "B.Sc Biotechnology", "ug", "Sciences", THREE_OR_FOUR,
           "UGC undergraduate framework",
           "Biology in class 12 (some universities also accept Mathematics students)",
           {"biology": 3, "health": 2, "data": 1},
           (("Year 1: Foundations", ("Cell Biology", "Biochemistry",
             "Chemistry for Life Sciences", "Biostatistics")),
            ("Year 2: Core", ("Microbiology", "Genetics", "Molecular Biology", "Immunology")),
            ("Year 3: Applied", ("Genetic Engineering", "Bioinformatics",
             "Industrial Biotechnology", "Project"))),
           needs_all=("Biology",)),
    Course("mbbs", "MBBS (Bachelor of Medicine and Bachelor of Surgery)", "ug", "Health",
           "5.5 years including a one-year internship",
           "National Medical Commission competency-based curriculum",
           "Physics, Chemistry, Biology and English in class 12, with NEET-UG",
           {"health": 3, "biology": 3, "people": 1},
           (("Phase 1: Pre-clinical", ("Anatomy", "Physiology", "Biochemistry")),
            ("Phase 2: Para-clinical", ("Pathology", "Pharmacology", "Microbiology",
             "Forensic Medicine and Toxicology")),
            ("Phase 3: Clinical", ("Community Medicine", "Ophthalmology",
             "Otorhinolaryngology", "General Medicine", "General Surgery",
             "Obstetrics and Gynaecology", "Paediatrics")),
            ("Internship", ("Compulsory rotating internship",))),
           needs_all=("Physics", "Chemistry", "Biology")),
    Course("bpharm", "B.Pharm (Bachelor of Pharmacy)", "ug", "Health", "4 years",
           "Pharmacy Council of India syllabus",
           "Physics and Chemistry with Mathematics or Biology in class 12",
           {"health": 3, "biology": 2},
           (("Year 1", ("Human Anatomy and Physiology", "Pharmaceutical Analysis",
             "Pharmaceutics", "Pharmaceutical Inorganic Chemistry")),
            ("Year 2", ("Pharmaceutical Organic Chemistry", "Physical Pharmaceutics",
             "Pharmaceutical Microbiology", "Pharmaceutical Engineering")),
            ("Year 3", ("Medicinal Chemistry", "Industrial Pharmacy", "Pharmacology",
             "Pharmacognosy")),
            ("Year 4", ("Novel Drug Delivery Systems", "Biopharmaceutics and Pharmacokinetics",
             "Project"))),
           needs_all=("Physics", "Chemistry"), needs_any=("Mathematics", "Biology")),
    Course("bcom", "B.Com", "ug", "Commerce and management", THREE_OR_FOUR,
           "UGC undergraduate framework",
           "Commerce is preferred; many universities accept any stream, some ask for Mathematics",
           {"finance": 3, "business": 2, "data": 1},
           (("Year 1: Foundations", ("Financial Accounting", "Business Economics",
             "Business Law", "Business Statistics")),
            ("Year 2: Core", ("Corporate Accounting", "Cost Accounting", "Income Tax",
             "Company Law")),
            ("Year 3: Advanced", ("Auditing", "Financial Management",
             "Management Accounting")))),
    Course("bba", "BBA (Bachelor of Business Administration)", "ug", "Commerce and management",
           THREE_OR_FOUR, "University syllabi", "Any stream",
           {"business": 3, "people": 1, "finance": 1},
           (("Year 1: Foundations", ("Principles of Management", "Business Economics",
             "Financial Accounting", "Business Communication")),
            ("Year 2: Core", ("Marketing Management", "Human Resource Management",
             "Organisational Behaviour", "Business Statistics")),
            ("Year 3: Advanced", ("Financial Management", "Strategic Management",
             "Entrepreneurship", "Internship and Project")))),
    Course("ba_psych", "B.A. Psychology", "ug", "Humanities and social sciences", THREE_OR_FOUR,
           "UGC undergraduate framework", "Any stream",
           {"people": 3, "health": 1, "society": 1},
           (("Year 1: Foundations", ("Introduction to Psychology", "Biopsychology",
             "Statistics for Psychology", "Developmental Psychology")),
            ("Year 2: Core", ("Social Psychology", "Cognitive Psychology", "Research Methods",
             "Personality")),
            ("Year 3: Applied", ("Abnormal Psychology", "Counselling Psychology",
             "Organisational Psychology", "Dissertation")))),
    Course("ba_econ", "B.A. Economics", "ug", "Humanities and social sciences", THREE_OR_FOUR,
           "UGC undergraduate framework",
           "Any stream; Mathematics in class 12 helps and is required by some universities",
           {"society": 3, "data": 2, "maths": 1, "finance": 1},
           (("Year 1: Foundations", ("Introductory Microeconomics",
             "Introductory Macroeconomics", "Mathematical Methods for Economics",
             "Statistics for Economics")),
            ("Year 2: Core", ("Intermediate Microeconomics", "Intermediate Macroeconomics",
             "Indian Economy", "Econometrics")),
            ("Year 3: Advanced", ("Development Economics", "International Economics",
             "Public Economics")))),
    Course("ba_english", "B.A. English", "ug", "Humanities and social sciences", THREE_OR_FOUR,
           "University syllabi", "Any stream", {"media": 2, "people": 1, "teaching": 1},
           (("Year 1: Foundations", ("British Poetry and Drama", "Indian Classical Literature",
             "Academic Writing")),
            ("Year 2: Core", ("American Literature", "Indian Writing in English",
             "Literary Criticism")),
            ("Year 3: Advanced", ("Postcolonial Literatures", "Literary Theory",
             "Women's Writing")))),
    Course("bjmc", "Bachelor of Journalism and Mass Communication (BJMC)", "ug", "Media",
           "3 years", "University syllabi", "Any stream", {"media": 3, "people": 1, "society": 1},
           (("Year 1: Foundations", ("Introduction to Mass Communication",
             "Reporting and Editing", "Media Writing")),
            ("Year 2: Core", ("Media Law and Ethics", "Photojournalism",
             "Radio and Television Production")),
            ("Year 3: Advanced", ("Digital Media", "Advertising and Public Relations",
             "Media Project")))),
    Course("ba_llb", "B.A. LL.B. (integrated law degree)", "ug", "Law", "5 years",
           "Bar Council of India rules",
           "Any stream, usually with an entrance exam such as CLAT",
           {"law": 3, "society": 2, "people": 1},
           (("Year 1", ("Legal Methods", "Law of Contract", "Political Science", "Sociology")),
            ("Year 2", ("Constitutional Law", "Law of Torts", "Family Law", "Economics")),
            ("Year 3", ("Criminal Law", "Property Law", "Jurisprudence",
             "Administrative Law")),
            ("Year 4", ("Company Law", "Labour Law", "Environmental Law",
             "Public International Law")),
            ("Year 5", ("Civil Procedure", "Law of Evidence", "Taxation Law",
             "Moot Court and Internship")))),
    Course("bdes", "B.Des (Bachelor of Design)", "ug", "Design and architecture", "4 years",
           "University syllabi",
           "Any stream, usually with a design entrance exam such as NID DAT or UCEED",
           {"design": 3, "media": 1, "business": 1},
           (("Year 1: Foundations", ("Design Fundamentals", "Drawing and Visualisation",
             "Colour and Form", "Materials and Processes")),
            ("Year 2: Core", ("Design Research", "Digital Design Tools",
             "Specialisation Studios")),
            ("Year 3: Advanced", ("Design Management", "Industry Internship")),
            ("Year 4: Graduation", ("Graduation Project", "Portfolio")))),
    Course("bsc_hha", "B.Sc Hospitality and Hotel Administration", "ug", "Hospitality",
           "3 years", "National Council for Hotel Management (NCHMCT)",
           "Any stream with English, usually through NCHM JEE",
           {"hospitality": 3, "business": 1, "people": 1},
           (("Year 1", ("Food Production", "Food and Beverage Service", "Front Office",
             "Housekeeping")),
            ("Year 2", ("Hotel Accountancy", "Nutrition", "Industrial Training")),
            ("Year 3", ("Hospitality Marketing", "Facility Planning",
             "Human Resource Management")))),
    Course("beled", "Bachelor of Elementary Education (B.El.Ed)", "ug", "Education", "4 years",
           "University syllabi", "Any stream", {"teaching": 3, "people": 2},
           (("Year 1", ("Child Development", "Contemporary India", "Nature of Language",
             "Core Mathematics")),
            ("Year 2", ("Cognition and Learning", "Human Relations and Communication",
             "Core Natural Science")),
            ("Year 3", ("Pedagogy of Language", "Pedagogy of Mathematics",
             "School Planning and Management")),
            ("Year 4", ("Curriculum Studies", "School Internship")))),

    # ------------------------------------------------------------- postgraduate
    Course("mtech_cse", "M.Tech Computer Science and Engineering", "pg",
           "Engineering and technology", "2 years", "AICTE",
           "A B.Tech or B.E. in Computer Science, IT or a closely related branch, usually "
           "through GATE", {"coding": 3, "maths": 2, "data": 2},
           (("Year 1", ("Advanced Algorithms", "Advanced Computer Architecture",
             "Machine Learning", "Research Methodology")),
            ("Year 2", ("Dissertation",))),
           degrees=frozenset({"btech_cs"})),
    Course("mca", "MCA (Master of Computer Applications)", "pg", "Computing", "2 years", "AICTE",
           "A degree such as BCA, B.Sc or B.Com with Mathematics in class 12 or in the degree, "
           "usually through NIMCET or a university test",
           {"coding": 3, "data": 1, "business": 1},
           (("Year 1", ("Data Structures and Algorithms", "Object-Oriented Programming",
             "Database Systems", "Operating Systems", "Discrete Mathematics")),
            ("Year 2", ("Software Engineering", "Computer Networks", "Cloud Computing",
             "Major Project"))),
           degrees=frozenset({"bca", "bsc_cs", "bsc_maths", "btech_cs", "btech_other", "bcom"})),
    Course("msc_ds", "M.Sc Data Science", "pg", "Computing", "2 years", "University syllabi",
           "A degree with Mathematics or Statistics, such as B.Sc, BCA or B.Tech",
           {"data": 3, "coding": 2, "maths": 2},
           (("Year 1", ("Probability and Statistics", "Programming for Data Science",
             "Machine Learning", "Database Systems")),
            ("Year 2", ("Deep Learning", "Big Data Analytics", "Data Visualisation",
             "Capstone Project"))),
           degrees=frozenset({"bsc_maths", "bsc_cs", "bca", "btech_cs", "btech_other"})),
    Course("msc_maths", "M.Sc Mathematics", "pg", "Sciences", "2 years", "UGC framework",
           "A B.Sc with Mathematics, usually through IIT JAM or CUET-PG",
           {"maths": 3, "data": 2},
           (("Year 1", ("Real Analysis", "Abstract Algebra", "Linear Algebra",
             "Ordinary Differential Equations", "Complex Analysis")),
            ("Year 2", ("Topology", "Functional Analysis", "Partial Differential Equations",
             "Project"))),
           degrees=frozenset({"bsc_maths"})),
    Course("msc_biotech", "M.Sc Biotechnology", "pg", "Sciences", "2 years", "University syllabi",
           "A degree in life sciences, biotechnology or pharmacy, usually through GAT-B or "
           "CUET-PG", {"biology": 3, "health": 2},
           (("Year 1", ("Cell and Molecular Biology", "Genetic Engineering", "Immunology",
             "Bioinformatics")),
            ("Year 2", ("Bioprocess Engineering", "Genomics and Proteomics",
             "Research Project"))),
           degrees=frozenset({"bsc_life", "bpharm"})),
    Course("mba", "MBA (Master of Business Administration)", "pg", "Commerce and management",
           "2 years", "AICTE and university syllabi",
           "Any bachelor's degree, usually through CAT, XAT, MAT or CMAT",
           {"business": 3, "finance": 2, "people": 1},
           (("Year 1", ("Managerial Economics", "Financial Accounting", "Marketing Management",
             "Organisational Behaviour", "Business Statistics", "Operations Management")),
            ("Year 2", ("Strategic Management", "Summer Internship",
             "Electives in Finance, Marketing, HR or Analytics", "Capstone Project")))),
    Course("mcom", "M.Com", "pg", "Commerce and management", "2 years", "UGC framework",
           "A B.Com, and at some universities a BBA or B.A. Economics, usually through CUET-PG",
           {"finance": 3, "business": 2},
           (("Year 1", ("Advanced Financial Accounting", "Corporate Finance",
             "Managerial Economics", "Research Methodology")),
            ("Year 2", ("International Business", "Advanced Taxation", "Financial Markets",
             "Dissertation"))),
           degrees=frozenset({"bcom", "bba", "ba_econ"})),
    Course("ma_econ", "M.A. Economics", "pg", "Humanities and social sciences", "2 years",
           "UGC framework",
           "Usually a B.A. or B.Sc in Economics; some universities accept any degree with "
           "Mathematics, through CUET-PG", {"society": 3, "data": 2, "maths": 2},
           (("Year 1", ("Microeconomic Theory", "Macroeconomic Theory", "Econometrics",
             "Mathematical Economics")),
            ("Year 2", ("Development Economics", "International Trade", "Public Economics"))),
           degrees=frozenset({"ba_econ", "bsc_maths", "bcom"})),
    Course("ma_psych", "M.A. Psychology", "pg", "Humanities and social sciences", "2 years",
           "UGC framework", "Usually a B.A. or B.Sc with Psychology",
           {"people": 3, "health": 2},
           (("Year 1", ("Cognitive Psychology", "Research Methods and Statistics",
             "Personality Theories", "Psychopathology")),
            ("Year 2", ("Counselling Psychology", "Organisational Psychology", "Practicum",
             "Dissertation"))),
           degrees=frozenset({"ba_psych"})),
    Course("ma_english", "M.A. English", "pg", "Humanities and social sciences", "2 years",
           "UGC framework",
           "Usually a B.A. in English; many universities accept any degree, through CUET-PG",
           {"media": 2, "teaching": 2, "people": 1},
           (("Year 1", ("Literary Theory", "British Literature", "Indian Writing in English")),
            ("Year 2", ("Postcolonial Studies", "Linguistics", "Dissertation")))),
    Course("mjmc", "M.A. Journalism and Mass Communication", "pg", "Media", "2 years",
           "University syllabi", "Any bachelor's degree, usually through a university test",
           {"media": 3, "society": 1},
           (("Year 1", ("Communication Theory", "Reporting and Editing",
             "Media Research Methods")),
            ("Year 2", ("Digital Journalism", "Media Management",
             "Dissertation or Media Project")))),
    Course("llb", "LL.B. (3-year law degree after graduation)", "pg", "Law", "3 years",
           "Bar Council of India rules",
           "Any bachelor's degree, usually through a university entrance test",
           {"law": 3, "society": 2},
           (("Year 1", ("Law of Contract", "Constitutional Law", "Law of Torts", "Family Law")),
            ("Year 2", ("Criminal Law", "Property Law", "Jurisprudence",
             "Administrative Law")),
            ("Year 3", ("Civil Procedure", "Law of Evidence", "Company Law",
             "Moot Court and Internship")))),
    Course("bed", "B.Ed (Bachelor of Education)", "pg", "Education", "2 years", "NCTE norms",
           "Any bachelor's degree, usually with a minimum percentage", {"teaching": 3, "people": 1},
           (("Year 1", ("Childhood and Growing Up", "Learning and Teaching",
             "Pedagogy of a School Subject", "Assessment for Learning")),
            ("Year 2", ("Gender, School and Society", "Creating an Inclusive School",
             "School Internship")))),
)

MAX_RESULTS = 6
_BY_KEY = {c.key: c for c in COURSES}


def options() -> dict:
    return {
        "method": METHOD,
        "note": NOTE,
        "levels": [{"value": k, "label": v, "input": LEVEL_INPUT[k]} for k, v in LEVELS.items()],
        "stages": [{"value": k, "label": v, "has_course": k in COURSE_STAGES}
                   for k, v in STAGES.items()],
        "tracks": [{"value": k, "label": v} for k, v in TRACKS.items()],
        "skill_note": SKILL_NOTE,
        "nep_ug_exits": list(NEP_UG_EXITS),
        "next_levels": {k: list(v) for k, v in NEXT_LEVELS.items()},
        "streams": [{"value": k, "label": v, "core": STREAM_SUBJECTS[k]["core"],
                     "optional": STREAM_SUBJECTS[k]["optional"]} for k, v in STREAMS.items()],
        "subjects_note": SUBJECTS_NOTE,
        "degrees": [{"value": k, "label": v} for k, v in DEGREES.items()],
        "interests": [{"value": k, "label": v} for k, v in INTERESTS.items()],
    }


# What each class 12 stream opens up, in the student's own words.
STREAM_LEADS: dict[str, str] = {
    "pcm": "Engineering (B.Tech, B.E.), computer science, mathematics, statistics and architecture",
    "pcb": "Medicine (MBBS), dentistry, pharmacy, nursing, and biology and life sciences",
    "pcmb": "Both routes: engineering and computer science, or medicine and life sciences",
    "commerce": "Commerce, accountancy, business administration, economics and finance",
    "arts": "Psychology, economics, English, law, sociology, design and teaching",
}


def subjects_now(stage: str, stream: str | None = None) -> dict:
    """The subjects a student is studying at this stage, with free study links.

    Class 10 has one national set of subjects; class 12 depends on the stream. A student
    already on a course studies that course's own subjects, so this returns nothing and
    the weekly path takes over."""
    if stage not in STAGES:
        raise ValueError("unknown stage")
    if stage == "class_10":
        core, optional, label = CLASS10_SUBJECTS["core"], CLASS10_SUBJECTS["optional"], "Class 10"
        schooling = "class 10"
    elif stage == "class_12":
        if stream not in STREAMS:
            raise ValueError("choose your class 12 stream")
        core = STREAM_SUBJECTS[stream]["core"]
        optional = STREAM_SUBJECTS[stream]["optional"]
        label = f"Class 12, {STREAMS[stream]}"
        schooling = "class 12"
    else:
        return {"stage": stage, "label": STAGES[stage], "subjects": [], "optional": [], "note": ""}
    # school subjects, so NCERT, Khan Academy and YouTube rather than NPTEL and SWAYAM
    with_links = lambda names: [{"name": n, "links": study_links(n, schooling)} for n in names]
    return {"stage": stage, "label": label, "subjects": with_links(core),
            "optional": with_links(optional), "note": SUBJECTS_NOTE}


def next_after(stage: str, stream: str | None = None) -> dict:
    """What a student can study after this stage.

    Class 10 leads to a class 12 stream or to a diploma, so both are returned: the
    streams are choices of subjects, the levels are courses in the catalogue."""
    if stage not in STAGES:
        raise ValueError("unknown stage")
    levels = [{"value": lv, "label": LEVELS[lv], "input": LEVEL_INPUT[lv]}
              for lv in NEXT_LEVELS[stage]]
    streams = []
    if stage == "class_10":
        streams = [{"value": k, "label": v, "leads_to": STREAM_LEADS[k],
                    "core": STREAM_SUBJECTS[k]["core"], "optional": STREAM_SUBJECTS[k]["optional"]}
                   for k, v in STREAMS.items()]
    return {"stage": stage, "label": STAGES[stage], "levels": levels, "streams": streams,
            "leads_to": STREAM_LEADS.get(stream or "", "")}


def _fit(score: int, matched: list, chosen: list) -> str:
    """How well a course answers what the student picked.

    Absolute thresholds made every course a "Possible fit" when only one interest was
    chosen, which told the student nothing. This compares the course against the
    interests they actually picked."""
    if len(matched) < len(chosen):
        return "Partial match"
    return "Strong match" if score >= 2 * len(chosen) else "Good match"


def recommend(level: str = "ug", interests: list[str] | None = None, stream: str | None = None,
              subjects: list[str] | None = None, degree: str | None = None) -> dict:
    """Rank courses at one level by how well they match the student's interests.

    A course is listed under 'courses' only if the student usually qualifies for it:
    by class 12 subjects for diplomas after class 12 and undergraduate degrees, by
    bachelor's degree for postgraduate study. Courses that match the interests but not
    the eligibility rules go under 'also_consider' with what they require, so no option
    is hidden.
    """
    if level not in LEVELS:
        raise ValueError("unknown level")
    chosen = [i for i in dict.fromkeys(interests or []) if i in INTERESTS]
    if not chosen:
        raise ValueError("choose at least one interest")

    taken: set[str] = set()
    needs = LEVEL_INPUT[level]
    if needs == "stream":
        if stream not in STREAMS:
            raise ValueError("choose your class 12 stream")
        allowed = STREAM_SUBJECTS[stream]
        taken = set(allowed["core"]) | {s for s in (subjects or []) if s in allowed["optional"]}
    if needs == "degree" and degree not in DEGREES:
        raise ValueError("choose your bachelor's degree")

    courses, also = [], []
    for course in COURSES:
        if course.level != level:
            continue
        matched = [i for i in chosen if course.interests.get(i)]
        score = sum(course.interests[i] for i in matched)
        if not score:
            continue
        labels = [INTERESTS[i] for i in matched]
        reason = "Matches your interests: " + ", ".join(labels) + "."
        # how much of the course itself is about what they picked, so that a course
        # centred on the interest outranks a course that merely touches it
        share = round(score / sum(course.interests.values()), 4)
        if course.eligible(taken, degree if needs == "degree" else None):
            courses.append({**course.to_dict(), "score": score, "share": share,
                            "matched": labels, "reason": reason,
                            "fit": _fit(score, matched, chosen)})
        else:
            also.append({"key": course.key, "name": course.name, "score": score,
                         "share": share, "matched": labels, "reason": reason,
                         "needs": course.eligibility})

    order = lambda c: (-c["score"], -c["share"], c["name"])
    return {
        "method": METHOD,
        "note": NOTE,
        "level": level,
        "level_label": LEVELS[level],
        "stream": stream if needs == "stream" else None,
        "stream_label": STREAMS.get(stream, "") if needs == "stream" else "",
        "subjects": sorted(taken),
        "degree": degree if needs == "degree" else None,
        "degree_label": DEGREES.get(degree, "") if needs == "degree" else "",
        "interests": [INTERESTS[i] for i in chosen],
        "courses": sorted(courses, key=order)[:MAX_RESULTS],
        "also_consider": sorted(also, key=order)[:3],
    }


def explore(level: str | None = None) -> dict:
    """Every course, grouped by level and category, with no eligibility filtering.

    For browsing beyond one's own background: a B.Tech student can see what an economics
    or law course involves, and follow the free study links for any subject."""
    if level is not None and level not in LEVELS:
        raise ValueError("unknown level")
    groups: dict[str, dict[str, list[dict]]] = {}
    for course in COURSES:
        if level and course.level != level:
            continue
        groups.setdefault(course.level, {}).setdefault(course.category, []).append(
            course.to_dict())
    return {
        "method": METHOD,
        "note": NOTE,
        "levels": [{"level": lv, "label": LEVELS[lv],
                    "categories": [{"category": cat, "courses": sorted(cs, key=lambda c: c["name"])}
                                   for cat, cs in sorted(groups[lv].items())]}
                   for lv in LEVELS if lv in groups],
        "count": sum(len(cs) for g in groups.values() for cs in g.values()),
    }


def course(key: str) -> dict:
    if key not in _BY_KEY:
        raise KeyError(key)
    return _BY_KEY[key].to_dict()


def exists(key: str) -> bool:
    return key in _BY_KEY


def looking_ahead(stream: str, level: str = "ug") -> dict:
    """Courses a class 10 student could take later, if they pick this stream.

    Read-only: they are two years away from applying, so this is what the stream opens
    up rather than a recommendation. Eligibility uses the stream's core subjects only,
    so nothing here depends on optional subjects they have not chosen yet."""
    if stream not in STREAMS:
        raise ValueError("unknown class 12 stream")
    if level not in LEVELS:
        raise ValueError("unknown level")
    taken = set(STREAM_SUBJECTS[stream]["core"])
    courses = [c.to_dict() for c in COURSES
               if c.level == level and c.eligible(taken, None)]
    return {
        "stream": stream, "stream_label": STREAMS[stream], "level": level,
        "level_label": LEVELS[level], "leads_to": STREAM_LEADS[stream],
        "courses": sorted(courses, key=lambda c: c["name"]),
        "note": ("What this stream opens up after class 12, using its core subjects only. "
                 "Optional subjects you choose in class 11 can open up more."),
    }


# The shortest and longest plans offered. Below four weeks a degree's subject list is
# meaningless; beyond three years a plan stops being a plan.
PLAN_WEEKS = (4, 8, 12, 16, 24, 36, 52, 78, 104, 156)
PLAN_NOTE = ("A study plan, not the official course timetable. Subjects keep their own "
             "order, so earlier years come first.")


def _monday_of(day: str) -> date:
    """The Monday of that week. A plan that starts mid-week reads as a short first
    week, which is not what "finish in 8 weeks" means to anyone."""
    try:
        chosen = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError("give the start date as YYYY-MM-DD") from exc
    return chosen - timedelta(days=chosen.weekday())


def weekly_plan(key: str, weeks: int = 24, start: str | None = None) -> dict:
    """Spread a course's subjects over however many weeks the student wants.

    Subjects are dealt out in order, so the load per week is as even as the list allows:
    26 subjects over 12 weeks gives weeks of 3 and weeks of 2, never 5 and then 1. Weeks
    are never empty, so asking for more weeks than there are subjects is refused.

    With a start date the weeks get real dates, Monday to Sunday, which is what makes a
    plan something you can put in a calendar and be late for."""
    course = _BY_KEY.get(key)
    if course is None:
        raise KeyError(key)
    if weeks < 1:
        raise ValueError("choose at least one week")
    subjects = [(year, name) for year, names in course.years for name in names
                if not _NOT_A_TOPIC.search(name)]
    if not subjects:
        raise ValueError("this course has no subjects to plan")
    weeks = min(int(weeks), len(subjects))

    begin = _monday_of(start) if start else None
    base, extra = divmod(len(subjects), weeks)
    out, at = [], 0
    for index in range(weeks):
        take = base + (1 if index < extra else 0)
        block = subjects[at:at + take]
        at += take
        week = {
            "week": index + 1,
            "years": sorted({year for year, _ in block}),
            "subjects": [{"name": name, "links": study_links(name)} for _, name in block],
        }
        if begin is not None:
            first = begin + timedelta(weeks=index)
            week["starts"] = first.isoformat()
            week["ends"] = (first + timedelta(days=6)).isoformat()
        out.append(week)
    return {
        "key": course.key, "name": course.name, "level_label": LEVELS[course.level],
        "weeks": out, "n_weeks": weeks, "n_subjects": len(subjects),
        "starts": out[0].get("starts"), "ends": out[-1].get("ends"),
        "per_week": round(len(subjects) / weeks, 1),
        # The plan actually shown must be one of the choices, or the dropdown claims a
        # length the page is not showing and picking it changes nothing.
        "options": sorted({w for w in PLAN_WEEKS if w <= len(subjects)} | {weeks}),
        "note": PLAN_NOTE,
    }
