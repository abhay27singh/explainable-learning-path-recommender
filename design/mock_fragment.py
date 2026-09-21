"""The recommended dashboard, as one markup string rendered light, dark and phone."""

MOCK = '''
<div class="app">
  <div class="topbar">
    <div class="brand">
      <svg viewBox="0 0 32 32" width="24" height="24" aria-hidden="true"><rect width="32" height="32" rx="9" fill="var(--accent)"></rect><path d="M7 24 L16 16 L25 8" stroke="var(--surface)" stroke-width="2.4" fill="none" stroke-linecap="round"></path></svg>
      Learning Path
    </div>
    <nav class="tabs">
      <span class="tab on">My learning</span><span class="tab">Progress</span>
      <span class="tab">Course Finder</span><span class="tab">Record</span>
    </nav>
    <div class="topright">
      <span class="iconbtn" title="Theme">__THEMEICON__</span>
      <span class="who">Gursimar</span><span class="chip solid">Student</span>
    </div>
  </div>

  <div class="panes">
    <aside class="rail">
      <div class="railhead">This week</div>
      <div class="days">
        <span class="day"><b>M</b><i>2</i></span><span class="day"><b>T</b><i>1</i></span>
        <span class="day on"><b>W</b><i>3</i></span><span class="day"><b>T</b><i>0</i></span>
        <span class="day"><b>F</b><i>0</i></span><span class="day"><b>S</b><i>1</i></span>
        <span class="day"><b>S</b><i>0</i></span>
      </div>
      <div class="railhead">Computer Science</div>
      <div class="bar"><span style="width:11%"></span></div>
      <div class="muted">4 of 35 weeks · ends 12 Jun 2027</div>
      <div class="railhead">Jump to</div>
      <div class="raillist">
        <span class="railitem on">Week 5 · now</span>
        <span class="railitem">Week 6 · 28 Sep</span>
        <span class="railitem warn">Week 3 · retake</span>
      </div>
    </aside>

    <main class="work">
      <div class="greet">
        <div>
          <h2>Good morning, Gursimar</h2>
          <p class="muted">Week 5 runs 21 to 27 September. One quiz is waiting to be retaken.</p>
        </div>
        <span class="btn primary">Add to my calendar</span>
      </div>

      <div class="tiles">
        <div class="tile"><span class="muted">Weeks done</span><b>4 <em>/ 35</em></b><div class="bar sm"><span style="width:11%"></span></div></div>
        <div class="tile fill"><span>Day streak</span><b>4</b><span class="tiny">longest 6</span></div>
        <div class="tile"><span class="muted">Average score</span><b>64</b><span class="tiny warn">2 below the pass mark</span></div>
      </div>

      <div class="panel next">
        <div class="panelhead">
          <div><span class="eyebrow">Your next step</span><h3>Computer Science · Week 5</h3></div>
          <span class="chip good">Ready · 21 Sep</span>
        </div>
        <ul class="reasons">
          <li>Builds on Week 4 and Week 1, which you already have.</li>
          <li>Finishing it opens up Week 6.</li>
        </ul>
        <div class="row">
          <span class="btn primary">I studied this</span>
          <span class="btn">I passed the test</span>
          <span class="btn">I found it hard</span>
        </div>
        <div class="score">
          <span class="muted">Took a quiz?</span>
          <span class="field">out of 100</span>
          <span class="btn">Save score</span>
          <span class="tiny muted">40 or more counts as a pass</span>
        </div>
      </div>

      <div class="panel">
        <div class="panelhead"><h3>The weeks ahead</h3><span class="tiny muted">dates from your plan</span></div>
        <div class="steps">
          <div class="step on"><span class="num">5</span><div><b>Week 5</b><span class="tiny muted">21 to 27 Sep · ready</span></div><span class="chip">Now</span></div>
          <div class="step"><span class="num">6</span><div><b>Week 6</b><span class="tiny muted">28 Sep to 4 Oct</span></div><span></span></div>
          <div class="step"><span class="num">7</span><div><b>Week 7</b><span class="tiny muted">5 to 11 Oct · revise Week 3 first</span></div><span class="chip warn">34 / 100</span></div>
        </div>
      </div>
    </main>

    <aside class="side">
      <div class="panel">
        <h4>Record of study</h4>
        <p class="tiny muted">Weeks finished, marks and dates. Printable.</p>
        <div class="recordrow"><span>Weeks finished</span><b>4</b></div>
        <div class="recordrow"><span>Quizzes recorded</span><b>6</b></div>
        <div class="recordrow"><span>Since</span><b>2 Sep 2026</b></div>
        <span class="btn wide">Open record</span>
      </div>
      <div class="panel">
        <h4>Recent scores</h4>
        <div class="scorerow"><span>Week 4</span><div class="bar sm"><span style="width:88%"></span></div><b>88</b></div>
        <div class="scorerow"><span>Week 3</span><div class="bar sm"><span class="dim" style="width:34%"></span></div><b class="warn">34</b></div>
        <div class="scorerow"><span>Week 2</span><div class="bar sm"><span style="width:71%"></span></div><b>71</b></div>
      </div>
      <div class="panel">
        <h4>Saved courses</h4>
        <div class="saved"><b>MCA</b><span class="tiny muted">Postgraduate · 2 years</span></div>
        <div class="saved"><b>Data Analytics Associate</b><span class="tiny muted">Skill path · 6 months</span></div>
      </div>
    </aside>
  </div>
</div>
'''

SUN = ('<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" '
       'stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"></circle>'
       '<path d="M12 3v2M12 19v2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M3 12h2M19 12h2M5.6 18.4 7 17M17 7l1.4-1.4"></path></svg>')
MOON = ('<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"></path></svg>')

def mock(dark: bool) -> str:
    return MOCK.replace("__THEMEICON__", SUN if dark else MOON)
