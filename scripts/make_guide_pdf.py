"""Build the Finance Watchdog project guide PDF - written in plain English.

ASCII only in body text: ReportLab's built-in fonts draw unicode dashes and arrows as black boxes.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
OUT = ROOT / "Finance_Watchdog_Project_Guide.pdf"

INK = colors.HexColor("#12141C")
ACCENT = colors.HexColor("#2E8B6B")
WARN = colors.HexColor("#C2552F")
MUTED = colors.HexColor("#5C6070")
RULE = colors.HexColor("#C9CCD4")
PANEL = colors.HexColor("#F4F5F7")

S = getSampleStyleSheet()


def st(name, **kw):
    base = kw.pop("parent", S["BodyText"])
    return ParagraphStyle(name, parent=base, **kw)


Body = st("Body", fontSize=9.8, leading=14.8, spaceAfter=7, alignment=TA_LEFT,
          textColor=colors.HexColor("#1A1C24"))
Lead = st("Lead", parent=Body, fontSize=11, leading=16.5, textColor=INK)
H1 = st("H1", parent=S["Heading1"], fontSize=19, leading=23, spaceBefore=2, spaceAfter=10, textColor=INK)
H2 = st("H2", parent=S["Heading2"], fontSize=13.5, leading=17, spaceBefore=14, spaceAfter=6, textColor=INK)
H3 = st("H3", parent=S["Heading3"], fontSize=11, leading=14, spaceBefore=10, spaceAfter=4, textColor=ACCENT)
Small = st("Small", parent=Body, fontSize=8.5, leading=12.4, textColor=MUTED)
Cap = st("Cap", parent=Small, spaceBefore=3, spaceAfter=10)
Bullet = st("Bullet", parent=Body, leftIndent=12, bulletIndent=2, spaceAfter=4)
Code = ParagraphStyle("Code", parent=S["Code"], fontName="Courier", fontSize=7.6, leading=10.2,
                      textColor=colors.HexColor("#14202B"), backColor=PANEL,
                      borderPadding=(6, 6, 6, 6), leftIndent=2, spaceAfter=2)
QStyle = st("Q", parent=Body, fontSize=10, leading=14, spaceBefore=9, spaceAfter=2, textColor=INK)

story = []
h1 = lambda t: story.append(Paragraph(t, H1))          # noqa: E731
h2 = lambda t: story.append(Paragraph(t, H2))          # noqa: E731
h3 = lambda t: story.append(Paragraph(t, H3))          # noqa: E731


def p(t, style=Body):
    story.append(Paragraph(t, style))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, Bullet, bulletText="\u2022"))


def gap(h=6):
    story.append(Spacer(1, h))


def code(txt, caption=None):
    story.append(Preformatted(txt.rstrip("\n"), Code))
    story.append(Paragraph(caption, Cap) if caption else Spacer(1, 7))


def table(rows, widths, fs=8.4):
    data = [[Paragraph(f"<b>{c}</b>" if i == 0 else str(c),
                       st(f"t{fs}", parent=Body, fontSize=fs, leading=fs * 1.38, spaceAfter=0))
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), PANEL),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    story.append(t)
    story.append(Spacer(1, 9))


def callout(title, text, colour=ACCENT):
    inner = [[Paragraph(f'<font color="{colour.hexval()}"><b>{title}</b></font><br/>{text}',
                        st("co", parent=Body, fontSize=9.4, leading=14, spaceAfter=0))]]
    t = Table(inner, colWidths=[164 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.4, colour),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(t)
    story.append(Spacer(1, 10))


def figure(path, caption, width=150 * mm, max_h=180 * mm):
    f = Path(path)
    if not f.exists():
        p(f"[missing figure: {f.name}]", Small)
        return
    from PIL import Image as PILImage
    iw, ih = PILImage.open(f).size
    w, h = width, width * ih / iw
    if h > max_h:
        h, w = max_h, max_h * iw / ih
    img = Image(str(f), width=w, height=h)
    img.hAlign = "LEFT"
    story.append(KeepTogether([img, Paragraph(caption, Cap)]))


def later_pages(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(22 * mm, 282 * mm, 188 * mm, 282 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(22 * mm, 285 * mm, "Finance Watchdog - Project Guide")
    canvas.drawRightString(188 * mm, 285 * mm, "github.com/priyanshsoni096-blip/finance-watchdog")
    canvas.drawCentredString(105 * mm, 12 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#EDEBE4"))
    canvas.setFont("Helvetica-Bold", 34)
    canvas.drawString(24 * mm, 205 * mm, "Finance Watchdog")
    canvas.setFont("Helvetica", 13.5)
    canvas.setFillColor(colors.HexColor("#9AA0AE"))
    canvas.drawString(24 * mm, 193 * mm, "Teaching computers to spot market cheating")
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2.5)
    canvas.line(24 * mm, 185 * mm, 62 * mm, 185 * mm)
    canvas.setFont("Helvetica", 10.5)
    canvas.setFillColor(colors.HexColor("#C9CCD4"))
    for i, ln in enumerate([
            "A complete guide in plain English: what problem this solves,",
            "how it is built, what goes in, what came out, the code that",
            "matters, the screens, and how to run and explain it."]):
        canvas.drawString(24 * mm, 172 * mm - i * 6.2 * mm, ln)
    canvas.setFillColor(colors.HexColor("#6E7380"))
    canvas.setFont("Helvetica", 9)
    canvas.drawString(24 * mm, 42 * mm, "Built on real NASDAQ order book data from 21 June 2012")
    canvas.drawString(24 * mm, 36 * mm, "214 automated tests - every number traced back to a saved file")
    canvas.setFillColor(WARN)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(24 * mm, 26 * mm, "SIMULATED DATA - this does not claim to catch real-world cheating")
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4, title="Finance Watchdog - Project Guide",
                      author="Finance Watchdog", leftMargin=22 * mm, rightMargin=22 * mm,
                      topMargin=26 * mm, bottomMargin=20 * mm)
doc.addPageTemplates([
    PageTemplate(id="cover", frames=[Frame(22 * mm, 20 * mm, 166 * mm, 253 * mm)], onPage=cover_page),
    PageTemplate(id="body", frames=[Frame(22 * mm, 20 * mm, 166 * mm, 253 * mm, id="body")],
                 onPage=later_pages)])
story += [NextPageTemplate("body"), PageBreak()]

# ───────────────────────────── contents + glossary ─────────────────────────────
h1("What is in this guide")
table([
    ["#", "Section", "The question it answers"],
    ["1", "The problem", "What is the cheating, and why is it hard to catch?"],
    ["2", "How it is built", "What are the parts and how do they fit together?"],
    ["3", "What goes in", "What data is used, and what does every number mean?"],
    ["4", "How it was done", "What were the steps, and what went wrong on the way?"],
    ["5", "What came out", "What did it prove, and what did it not?"],
    ["6", "The important code", "Which lines matter, and what does each one do?"],
    ["7", "The screens", "What am I looking at, and how do I drive it?"],
    ["8", "Running it", "Which commands, in what order?"],
    ["9", "Who built what", "Who did which half, and how they were merged"],
    ["10", "Questions", "What will I be asked, and what do I say?"],
], [11 * mm, 48 * mm, 105 * mm])

h2("Words you will need")
p("Only nine. Everything else in this guide is plain English.")
table([
    ["Word", "What it means, simply"],
    ["Order book", "The live list of every buy and sell order waiting at each price. Like a queue board."],
    ["Spoofing", "Placing a big order you never intend to complete, to trick others about demand."],
    ["Agent", "A computer program that takes actions and learns from the results."],
    ["Reinforcement learning (RL)",
     "Training by reward instead of by examples. You never show it the right answer - you only say "
     "'that went well' or 'that went badly', and it works the rest out."],
    ["Episode", "One run of the simulation. Here: 200 decisions covering 2,000 market events."],
    ["Training run (or 'seed')",
     "Training the same program again from a different random starting point. We do it three times and "
     "report the spread, because a result that only happens once is luck, not a finding."],
    ["Recall", "Of all the real cheating, what share did we catch? 1.0 means we caught everything."],
    ["Precision", "Of everything we accused, what share was actually cheating? 1.0 means no false alarms."],
    ["F1 score", "Precision and recall combined into one number between 0 and 1. Higher is better."],
    ["False positive rate", "How often we accuse someone innocent. Lower is better."],
], [40 * mm, 124 * mm])

callout("The one thing to understand before anything else",
        "This project has two halves, and they ended very differently.<br/><br/>"
        "The <b>attacker half worked</b>: computer programs taught themselves to cheat, purely by chasing "
        "profit. Nobody told them how.<br/><br/>"
        "The <b>detector half did not</b>, in an interesting way. It got very good scores - but we then "
        "discovered the test was far too easy, because the answer was accidentally hidden in the data we "
        "were feeding it. We found that ourselves and published it. That honesty is the point of the "
        "project, not a weakness in it.")

# ───────────────────────────── 1. PROBLEM ─────────────────────────────
story.append(PageBreak())
h1("1. The problem")

h2("What is spoofing?")
p("Imagine you want to sell a share. Nobody is paying much for it right now. So you do this:")
bullets([
    "You place a <b>huge buy order</b> at a slightly lower price. You do not want it to go through.",
    "Everyone else sees a big buyer on the board and thinks demand is rising. The price drifts up.",
    "You <b>sell</b> into that higher price.",
    "You <b>cancel</b> your huge buy order before anyone can take you up on it.",
])
p("You never intended to buy anything. The big order was bait. That is spoofing, and it is illegal - it is "
  "banned by name in US law (Dodd-Frank Act, section 747).")

h2("So why not just catch it?")
p("Because the action on its own is completely normal. Placing a big order and cancelling it is something "
  "honest traders do all day long - they change their mind, the price moves, they manage their risk. What "
  "makes it illegal is the <b>intention</b>, and no computer file records what someone intended.")
p("Exchanges today mostly use a simple rule: \"big order, cancelled quickly, flag it\". That does catch "
  "cheats. It also flags a huge number of innocent people - in our tests, <b>82 out of every 100</b> orders "
  "it flagged were legitimate.")

h2("What this project asked")
callout("The question",
        "Can a computer <b>learn</b> to spot market cheating it has never seen before, without accusing "
        "honest traders - and without anyone telling it what cheating looks like?")

p("There were three things to prove. Here is how each one ended:")
table([
    ["What we wanted to prove", "How it ended"],
    ["1. The programs really learn to cheat (they are not just exploiting a bug in our simulation)",
     "<b>Mostly yes.</b> They make real money in our main simulation. In a second, different simulation "
     "they lose money - so the result depends on our assumptions."],
    ["2. The detector can catch cheating styles it has never seen",
     "<b>No at first, then yes</b> - but we later found the test was too easy, so the 'yes' means less "
     "than it looks."],
    ["3. The detector does not accuse innocent traders",
     "<b>Yes.</b> It wrongly flags 0.2 out of 100, where the simple rule flags 82 out of 100."],
], [82 * mm, 82 * mm])

# ───────────────────────────── 2. ARCHITECTURE ─────────────────────────────
story.append(PageBreak())
h1("2. How it is built")

p("Four parts. Think of it as a stage with actors on it.", Lead)
table([
    ["Part", "Think of it as", "What it actually is"],
    ["<b>1. The market</b>", "The stage",
     "A real day of NASDAQ trading, replayed order by order, with a model that makes the price move when "
     "our agent trades."],
    ["<b>2. The cheaters</b>", "The villains",
     "Programs told only one thing: make money. Never told what spoofing is. If they start spoofing, it is "
     "because they worked out that it pays."],
    ["<b>3. The watchdog</b>", "The detective",
     "A program that watches one trader and says 'suspicious' or 'fine' at every moment. Never given the "
     "rule for what spoofing looks like."],
    ["<b>4. The simple rule</b>", "The old guard",
     "\"Big order, cancelled fast.\" This is roughly what real exchanges use today. It is the thing we have "
     "to beat."],
], [26 * mm, 26 * mm, 112 * mm])

h2("How information flows")
code("""  Real NASDAQ data  ->  Simulated market  ->  Cheater learns to make money
                               |                        |
                               |                        v
                               |              Recordings of what it did
                               |                        |
                               v                        v
                     Price-impact model        Watchdog learns to spot it
                     (how much the price                |
                      moves when we trade)              v
                                              Its answers are compared with
                                              the simple rule, on the SAME
                                              recordings - a fair test""",
     "The cheaters are frozen before the watchdog starts learning. That way the cheaters cannot adapt to the "
     "detective, and the comparison stays fair.")

h2("Why there are two different markets")
p("Any result that only appears in one simulation is not really a result about cheating - it is a result "
  "about that simulation. So everything is checked twice.")
table([
    ["Market", "How the price moves", "What it is for"],
    ["<b>Replay market</b>", "Real historical prices, plus a model of how our own orders push the price",
     "The main results."],
    ["<b>Synthetic market</b>", "Prices appear naturally from simple simulated traders buying and selling "
     "from each other", "A second opinion. Built to match real statistics, never tuned to make spoofing pay."],
], [34 * mm, 68 * mm, 62 * mm])

callout("An important honest result",
        "Spoofing <b>makes money</b> in the first market and <b>loses money</b> in the second one, where a "
        "fake order moves the price 4 to 9 times less. We do not know which simulation is closer to reality. "
        "So the correct statement is: spoofing pays <i>under our price model</i> - not 'spoofing pays'.", WARN)

# ───────────────────────────── 3. INPUTS ─────────────────────────────
story.append(PageBreak())
h1("3. What goes in")

h2("The raw data")
table([
    ["Thing", "Value"],
    ["Where it comes from", "LOBSTER, a public dataset of real order book activity"],
    ["Which market and day", "NASDAQ, 21 June 2012 - <b>one single day</b> (this limits everything)"],
    ["Which companies", "Apple, Microsoft, Google, Intel, Amazon"],
    ["How much", "About 2.1 million individual order messages"],
    ["What we throw away", "The first 30,000 messages. The market is strange right after it opens - Intel's "
     "order book was 12 times thinner than usual."],
    ["One 'episode'", "200 decisions by the agent, covering 2,000 market events"],
], [42 * mm, 122 * mm])

h2("How much of it was used")
table([
    ["Where", "How many episodes", "What it is for"],
    ["Training", "150 per source, about 16 sources", "Teaching the programs"],
    ["Testing", "50 per source", "Choosing settings and checking progress"],
    ["Held out", "50 per source", "Never used for any decision - the real exam"],
    ["Each new attack test", "50 per company, all five companies", "Run once, result committed in advance"],
    ["Profit measurements", "50 episodes, 3 training runs", "Checking the money is real"],
], [34 * mm, 54 * mm, 76 * mm])

h2("What the detective sees: 46 numbers, every moment")
p("This is the most important part of the whole project, so it is worth going slowly.")
p("At every step, the watchdog is handed 46 numbers. <b>Forty</b> of them describe the public order book - "
  "what anyone watching the market could see. <b>Six</b> describe the specific trader being watched. Two of "
  "those six turned out to be a problem.")

h3("Numbers 0 to 39: the public order book")
p("Ten price levels on each side. For each level we record the price and the size of orders waiting there.")
table([
    ["Numbers", "What it is", "Why it is written this way"],
    ["prices", "How far this price sits from the middle of the market",
     "Measured in 'spreads' rather than dollars, so a $585 share and a $27 share can be compared fairly."],
    ["sizes", "How many shares are waiting at this price",
     "Compared against how deep the book normally is, so 'a big order' means the same thing on a busy "
     "stock and a quiet one."],
], [26 * mm, 62 * mm, 76 * mm])
p("The watched trader's own orders are mixed into these sizes, exactly as a real market feed would show "
  "them. This block does not say <i>who</i> placed what.", Small)

h3("Numbers 40 to 45: the trader being watched")
p("These only exist because our simulation knows which orders belong to whom. A real anonymous market feed "
  "could not give you these.")
table([
    ["Number", "What it tells you", "Does it give away the answer?"],
    ["40", "How big their resting order is", "A little - it says a big order exists"],
    ["41", "How long their oldest order has been sitting", "No"],
    ["42", "How much they just placed", "No"],
    ["43", "How much they just cancelled", "No"],
    ["44", "<b>Did they just buy or sell?</b> (+1 bought, -1 sold, 0 neither)", "<b>Yes - together with 45</b>"],
    ["45", "<b>Which side is their big order on?</b> (buy side or sell side)", "<b>Yes - together with 44</b>"],
], [18 * mm, 86 * mm, 60 * mm])

h2("What we are trying to predict")
p("We mark an order as cheating if, <b>while that order was sitting there, its owner traded the other way</b>. "
  "That is our definition of spoofing for this project.")

callout("Now look again at numbers 44 and 45",
        "Our definition is: <i>\"traded the other way while their order was sitting there\"</i>.<br/><br/>"
        "Number 44 says <b>which way they traded</b>. Number 45 says <b>which side their order is on</b>.<br/>"
        "<br/>Put those two together and you have simply restated the definition. So we tried it: just "
        "compare those two numbers, with <b>no learning at all</b>. It scores <b>0.997 out of 1.0</b>.<br/>"
        "<br/>In other words, we had accidentally been handing the detective the answer sheet.", WARN)

# ───────────────────────────── 4. HOW ─────────────────────────────
story.append(PageBreak())
h1("4. How it was done, step by step")

h2("Step 1 - make the market push back")
p("Historical prices do not react to someone who was not there. If our fake order cannot move the price, "
  "spoofing cannot work and there is nothing to study.")
p("So we measured, from the real data, <b>how much the price actually moves</b> when buying and selling gets "
  "lopsided. That measurement is then used to move the price when our agent trades. It is measured per "
  "company, not guessed.")
p("We tried a second way of measuring this and <b>threw it away</b>: it implied the price would jump 44 to "
  "58 spreads, which is not believable. We kept the rejected method in the project as a record.", Small)

h2("Step 2 - let the cheating appear by itself")
p("The cheater programs get one instruction: make money. They have six possible actions - buy, sell, place a "
  "big buy order, place a big sell order, cancel, do nothing. The word 'spoofing' appears nowhere in their "
  "training.")
p("Cheating appeared on Microsoft and Intel only. On Apple, Google and Amazon the gap between buy and sell "
  "prices is too wide for it to pay, so those programs never bothered. That is a finding in itself.")

figure(ROOT / "results" / "figures" / "fig1_spoofer_training.png",
       "A cheating program learning, over 600,000 practice steps. Profit (left) climbs as it discovers the "
       "trick; the right-hand chart shows it placing more and more large fake orders as it learns they pay.",
       width=152 * mm)

h3("Four times the simulation was cheated instead")
p("Four times we saw big profits and got excited, and four times it turned out the program had found a hole "
  "in our simulation rather than a real strategy. Each was only caught by asking <i>where exactly</i> the "
  "money came from.")
table([
    ["What it found", "What we saw", "How we closed it"],
    ["Free shares", "Buying far below the real market price; $253,000 a go",
     "Made the agent's own orders move the price against it too"],
    ["Swinging back and forth", "Tiny cost, unlimited benefit; $240,000 a go",
     "Price impact now builds up over time, and the benefit is capped at the order's size"],
    ["Quitting while behind", "All 31 early exits happened right before a loss",
     "No early exits allowed"],
    ["Trading at the open", "The order book is unusually thin just after opening",
     "Skip the first 30,000 messages"],
], [34 * mm, 68 * mm, 62 * mm])

h2("Step 3 - train the detective, and watch it fail")
p("The watchdog gets points for being right: <b>+1</b> for catching cheating, <b>-1</b> for missing it, "
  "<b>-2</b> for a false accusation, <b>0</b> for correctly staying quiet. The -2 is deliberate: wrongly "
  "accusing someone is worse than missing someone.")
p("On cheating styles it had seen before, it did well. On <b>new</b> styles it failed badly - it caught 22 "
  "out of 100 where the old simple rule caught 89.")

h2("Step 4 - work out why")
callout("The moment the project turned around",
        "The detective was not broken. <b>Our labelling was.</b><br/><br/>"
        "We had been marking the <i>entire time</i> a cheating order sat there as 'cheating happening now' - "
        "including the period <b>before</b> the person had actually done anything wrong.<br/><br/>"
        "So the detective learned the wrong job. Instead of learning <i>'this is cheating happening'</i>, it "
        "learned <i>'this person looks like a cheater'</i>. The proof: it flagged <b>99%</b> of one cheater's "
        "orders <i>before</i> that cheater had done anything, and caught only <b>14%</b> of another cheater's "
        "actual cheating moments.")

p("Three changes fixed it:")
bullets([
    "<b>Mark the act, not the person.</b> Only start calling it cheating from the moment they actually trade "
    "the other way.",
    "<b>Judge the moment in front of you</b> rather than trying to predict the future.",
    "<b>Show it cheaters with different timing</b> during training, so it cannot rely on one rhythm.",
])
p("Catching new cheating styles went from <b>22 out of 100</b> to <b>98 out of 100</b>.")

h2("Step 5 - check whether we had actually achieved anything")
p("We had not, as much as it looked. Two checks showed why:")
bullets([
    "Just comparing numbers 44 and 45 - <b>no learning whatsoever</b> - scores 0.997. Our clever detective "
    "reached a bar that needs no cleverness at all.",
    "A much simpler standard learning method matches it, and trains in <b>minutes</b> instead of hours.",
])

# ───────────────────────────── 5. RESULTS ─────────────────────────────
story.append(PageBreak())
h1("5. What came out")

h2("The cheaters - this part genuinely worked")
table([
    ["Program", "Company", "Profit per episode", "With price-impact turned off", "In the other market"],
    ["SPOOFER-04", "Intel", "<b>+$14,368</b>", "-$8,279", "-$3,270"],
    ["SPOOFER-02", "Microsoft", "<b>+$11,431</b>", "-$7,037", "-$2,930"],
    ["The other three", "Apple, Google, Amazon", "about zero - never learned to cheat", "-", "-"],
], [26 * mm, 30 * mm, 36 * mm, 36 * mm, 32 * mm, ], fs=8.0)
p("<b>Read the fourth column.</b> It is the proof. If we switch off the part of the simulation that lets "
  "fake orders move the price, the exact same trading <b>loses money</b>. So the profit comes from the "
  "trickery, not from being good at trading.", Small)
p("Each result is from three separate training runs. Microsoft's result swings a lot between runs (+$3,628 "
  "to +$15,886). We report that rather than hiding it behind an average.", Small)

figure(ROOT / "results" / "figures" / "fig5_seed_robustness.png",
       "Each cheating program trained three separate times. Intel (left) lands in the same place every time. "
       "Microsoft (right) swings a lot between runs - we show that rather than hiding it behind an average.",
       width=152 * mm)

h2("The detective - three versions")
table([
    ["What we measured", "First try", "After the fix", "Final version", "Old simple rule"],
    ["Catching cheating it had seen before", "0.75", "0.98", "<b>0.98</b>", "0.35"],
    ["Wrongly accusing innocent traders", "3 in 100", "0.3 in 100", "<b>0.2 in 100</b>", "82 in 100"],
    ["Catching new style 1 (late burst)", "22 in 100", "48 in 100", "<b>98 in 100</b>", "89 in 100"],
    ["Catching new style 2 (slow drip)", "7 in 100", "39 in 100", "<b>99 in 100</b>", "74 in 100"],
    ["Catching new style 3 (reversal)", "14 in 100", "45 in 100", "<b>98 in 100</b>", "96 in 100"],
    ["Catching new style 4 (layering)", "30 in 100", "-", "<b>99 in 100</b>", "100 in 100"],
], [52 * mm, 24 * mm, 24 * mm, 28 * mm, 28 * mm], fs=8.0)

figure(ROOT / "results" / "figures" / "fig3_detection_comparison.png",
       "The detective against the old rule across every test. Tall green is good on the left-hand charts, "
       "short is good on the right-hand one (false accusations).", width=152 * mm)

h2("But here is the ceiling")
p("Everything above has to be read next to this table. These are all scored on the same data.")
table([
    ["Method", "Score (out of 1.0)", "How long it takes to build"],
    ["Just compare numbers 44 and 45", "<b>0.997</b>", "No training at all"],
    ["Write the definition out as a rule", "0.986", "No training at all"],
    ["A standard learning method", "0.986", "A few minutes"],
    ["Our reinforcement learning detective", "0.984", "A million training steps"],
    ["A simple statistical model", "0.528", "Seconds"],
    ["The old 'big and cancelled' rule", "0.352", "Two settings"],
], [68 * mm, 44 * mm, 52 * mm])
p("Our clever method is <i>fourth</i>, behind something that requires no learning whatsoever. That is the "
  "honest headline.", Small)

figure(ROOT / "results" / "figures" / "fig6_pnl_suppression.png",
       "What happens to the cheater's profit when surveillance is switched on inside the simulation. The "
       "detective removes more of it than the old rule does - and bothers innocent traders far less often.",
       width=140 * mm)

h2("The fair test: take away the answer sheet")
p("So we removed numbers 40 to 45 entirely and tried to detect cheating using <b>only the public order "
  "book</b> - what a real outside observer would actually have.")
table([
    ["Method (public data only)", "Caught", "False alarms", "Score"],
    ["Flag absolutely everything", "100 in 100", "100 in 100", "0.218"],
    ["Simple book-imbalance rule", "9 in 100", "9 in 100", "0.099"],
    ["<b>Simple statistical model</b>", "38 in 100", "8 in 100", "<b>0.392</b>"],
    ["Pattern-finding model", "13 in 100", "2 in 100", "0.203"],
    ["Sequence-learning model", "3 in 100", "1 in 100", "0.052"],
], [58 * mm, 32 * mm, 36 * mm, 38 * mm])
p("<b>0.392, down from 0.997.</b> That is the honest size of the problem once you stop handing over the "
  "answer. It still beats flagging everything (0.218), so there is a real signal in the public data - just a "
  "modest one.")

callout("Three things this table hides unless you look carefully",
        "<b>1.</b> The most sophisticated model is the <i>best</i> on familiar data and the <i>worst</i> on "
        "new data - it memorised instead of learning. If we had only reported familiar-data scores, it would "
        "have looked like the winner.<br/>"
        "<b>2.</b> It works on Intel and Microsoft and is <i>worse than flagging everything</i> on the other "
        "three companies. The single average hides two opposite results.<br/>"
        "<b>3.</b> None of the models' confidence levels are trustworthy on new data - they are worse than "
        "simply guessing the average rate.", WARN)

# ───────────────────────────── 6. CODE ─────────────────────────────
story.append(PageBreak())
h1("6. The important code")

p("Six pieces. Each is short, and each one does something the whole project depends on.", Lead)

h2("6.1 The definition of cheating - two lines")
code("""evaluation/baseline_detector.py, line 51

    @property
    def manipulative(self) -> bool:
        return self.opposite_trades > 0""")
p("<b>In words:</b> an order counts as cheating if, while it was sitting there, its owner traded the other "
  "way at least once. <font face='Courier'>opposite_trades</font> simply counts those trades. That is the "
  "whole definition - everything else in the project is measured against it.")

h2("6.2 The six numbers about the watched trader - where the problem lives")
code("""evaluation/rollout.py, line 37

    return np.array([
        np.log1p((bid + ask) / depth),                            # 40 how big their order is
        min(age / 200.0, 5.0),                                    # 41 how long it has been sitting
        np.log1p(placed / depth),                                 # 42 what they just placed
        np.log1p(cancelled / depth),                              # 43 what they just cancelled
        float(trade_dir),                                         # 44 DID THEY BUY OR SELL?
        np.sign(imbalance) * np.log1p(abs(imbalance) / depth),    # 45 WHICH SIDE IS THEIR ORDER?
    ], dtype=np.float32)""")
p("<b>In words:</b> line 44 records which way they traded. Line 45 records which side their big order is on. "
  "Our definition of cheating is 'traded the other way while their order was sitting there'. These two lines "
  "are that sentence, written as numbers.")

h2("6.3 The two-line detector that exposed the problem")
code("""scripts/label_observability.py, line 40

    "opposite-side signature":
        lambda X: (np.sign(X[:, RESTING_IMBALANCE]) * np.sign(X[:, TRADE_DIR])) < 0,""")
p("<b>In words:</b> <font face='Courier'>np.sign</font> turns a number into +1, -1 or 0. If the order side "
  "and the trade direction point <i>opposite</i> ways, multiplying them gives a negative number. So: "
  "<i>if negative, flag it</i>.")
p("There is nothing to train and no settings to choose. It scores <b>0.997</b>. This single line is why the "
  "project's detector results have to be described as a ceiling rather than a success.")

h2("6.4 The fix that changed everything")
code("""watchdog/dataset.py, line 156

    y = np.zeros(len(src["y"]), dtype=bool)          # start with 'nothing is cheating'
    off = src["ep_offsets"]
    for ep, o in orders_from_array(src["orders"]):   # look at every big order
        if not o.manipulative:
            continue                                 # skip the honest ones
        a = off[ep]
        trades = src["X"][a + o.placed_step:a + o.removed_step, TRADE_COL]
        hit = np.flatnonzero(trades == -o.side)      # first trade the OTHER way
        if hit.size:
            y[a + o.placed_step + hit[0]:a + o.removed_step] = True
    return y""")
p("<b>In words, line by line.</b> Start by assuming nothing is cheating. Go through every big order and "
  "ignore the honest ones. For a cheating order, look at what its owner did while it was sitting there. Find "
  "the <b>first</b> moment they traded the other way - that is "
  "<font face='Courier'>trades == -o.side</font>, where the minus sign means 'the opposite side'. Mark "
  "cheating from <b>that</b> moment, not from when the order was placed.")
p("<b>Why it mattered.</b> The old version marked the whole period, so the detective could score well just "
  "by recognising who the cheater was. The new version only gives credit for reacting once something has "
  "actually happened. It keeps 87% of the marks - the change is not <i>how many</i>, but <i>which ones</i>.")

h2("6.5 The old rule we had to beat")
code("""evaluation/baseline_detector.py, line 60

    def flags(self, order: OrderRecord) -> bool:
        return (order.depth_mult >= self.n_mult              # at least N times normal size
                and order.removed_by == "cancel"             # cancelled, not completed
                and order.lifetime_events <= self.m_events)  # within M market events""")
p("<b>In words:</b> big, cancelled, and quick. Two settings, chosen using training data only (at least 2x "
  "normal size, cancelled within 1,000 market events). It catches almost every cheat - and also flags 82 "
  "out of every 100 honest large orders, which is why no one could actually use it.")

h2("6.6 How the detective is scored")
code("""watchdog/watchdog_env.py, line 61

    if manipulative_order_resting:
        reward = self.tp_reward if flag else -self.fn_cost    # +1 caught it, -1 missed it
    else:
        reward = -self.fp_cost if flag else 0.0               # -2 false alarm, 0 stayed quiet""")
p("<b>In words:</b> a false accusation costs twice as much as a miss. That is on purpose - investigating an "
  "innocent trader is expensive and unfair, so we want the detective to keep quiet when it is unsure.")
p("<b>A failure worth knowing about.</b> At first the detective learned to <i>never</i> flag anything, "
  "because staying silent scored better than guessing. The cause was not the scoring - it was that the 46 "
  "numbers were on wildly different scales. Putting them on a common scale fixed it.", Small)

# ───────────────────────────── 7. UI ─────────────────────────────
story.append(PageBreak())
h1("7. The screens")

p("One app, three screens. Every number on screen is read from a saved results file. Nothing is calculated, "
  "trained or downloaded when you open it.", Lead)

code("streamlit run app.py --server.fileWatcherType none")
p("<b>That last part is not optional.</b> This project contains a folder called "
  "<font face='Courier'>watchdog</font>, and Streamlit happens to use a tool with the same name. Without the "
  "flag they clash and the app will not start. If anything goes wrong on the day, open the file "
  "<font face='Courier'>dashboard.html</font> instead - same numbers, no app needed.", Small)

h2("Screen 1 - Overview: the whole page")
figure(SHOTS / "01_overview_full.jpg",
       "The entire first screen. Read it top to bottom: what the project is, what it found (green bar), three "
       "cards about the detective, three about the cheaters, the honest warning in orange, then two charts "
       "and the main finding.", width=160 * mm)

p("<b>Why the cards are in two labelled groups.</b> The cheaters' profit must never be mistaken for the "
  "detective's performance. They are different claims, so they sit in separate rows with their own headings.")
p("<b>Every card has four parts:</b> the big number, the old rule's number underneath for comparison, a line "
  "telling you which direction is good, and the name of the file the number came from. That last part means "
  "anything on screen can be checked in one step.")

figure(SHOTS / "ui_overview_kpi.jpg",
       "A single card close up. 0.984 is ours, 0.352 is the old rule, 'higher is better' tells you how to "
       "read it, and posthoc_rescore.md is where it came from.", width=130 * mm)

story.append(PageBreak())
h2("Screen 2 - Demonstration: watching one cheat happen")
figure(SHOTS / "02_demonstration_full.jpg",
       "The whole demonstration screen. Along the top are the five stages of the project. Below that: pick a "
       "recording, jump to a moment, and step through it. The three panels show the order book, what is "
       "happening right now, and a running log.", width=160 * mm)

p("<b>How to drive it.</b> Choose a recording from the dropdown, then use the three buttons - <b>Start</b>, "
  "<b>Manipulation</b> (jumps to the first cheating moment) and <b>First flag</b> (jumps to when the "
  "detective reacts). The slider moves one step at a time.")
p("<b>There are three recordings, and you should show at least two:</b> a program that taught itself to "
  "cheat, a cheating style the detective had <i>never seen before</i>, and an <b>honest trader</b> who "
  "places big orders and cancels them without cheating. That last one is how you prove it does not accuse "
  "innocent people.")

figure(SHOTS / "ui_demo_ladder.jpg",
       "The order book close up. Red rows are people selling, green are people buying, and the number is the "
       "price. The wide orange bar is the watched trader's own fake order - many times bigger than anything "
       "around it. That is what spoofing looks like.", width=130 * mm)

callout("Say this sentence out loud when you demo",
        "\"This is a <b>recording</b> being replayed, not a live market.\" The screen says so too. Claiming "
        "it is live would be the quickest way to lose marks.", WARN)

story.append(PageBreak())
h2("Screen 3 - Evidence: how hard is the task?")
figure(SHOTS / "03_evidence_how_hard.jpg",
       "The most important screen in the project, and all four of its tables. Top: the no-learning-required "
       "method scored as if it were a detector. Then: is the fancy method needed (no). Then: every method "
       "against the ceiling. Bottom: what happens when the giveaway numbers are taken away.", width=160 * mm)

p("<b>Why this tab comes first.</b> You cannot judge our scores until you know what the bar is. Putting our "
  "own results first and this afterwards would be misleading, so we do it the other way round.")

figure(SHOTS / "ui_evidence_table.jpg",
       "The first table close up. 'train (fitted by nothing)' is not a typo - that row really is a method "
       "with nothing to train.", width=130 * mm)

story.append(PageBreak())
h2("Screen 3, second tab - the three versions of the detective")
figure(SHOTS / "04_evidence_generations.jpg",
       "The three detectives side by side, so you can see exactly what the labelling fix changed. Below the "
       "table is the explanation, and at the bottom are the six honest limitations.", width=160 * mm)

p("<b>Read the LATEBURST row.</b> 0.217, then 0.484, then 0.984. That single row is the story of the project: "
  "a failure, a diagnosis, and a fix.")

story.append(PageBreak())
h2("Screen 3, third tab - the tests we committed to in advance")
figure(SHOTS / "05_evidence_heldout.jpg",
       "The two tests we wrote down before creating their data. Note the orange badges: HYPOTHESIS NOT "
       "SUPPORTED. These are the results that went against us, and they are shown exactly as they came out.",
       width=160 * mm)

p("<b>This is the screen to be proud of.</b> We predicted the detective would beat the old rule on cheating "
  "it had never seen. It did not - it got 0.24 against the rule's 0.89. We published that unchanged. Later "
  "versions of the detective do much better, but those numbers appear on the previous tab, clearly labelled "
  "as coming after the fact.")

figure(SHOTS / "ui_main_finding.jpg",
       "The page ends with the main finding written out in full, and a red warning. Red is used for exactly "
       "one thing in this whole app - the reminder that all of it is simulated.", width=130 * mm)

h2("If a file is missing")
p("The app does not crash. The affected card simply reads <b>\"Not available\"</b> and everything else still "
  "works - we tested this on purpose. One thing to remember: the app loads the files once when it starts, so "
  "if you change a file you must <b>restart</b> it. Refreshing the page is not enough.")

# ───────────────────────────── 8. RUNNING ─────────────────────────────
story.append(PageBreak())
h1("8. Running it")

h2("First-time setup")
code("""pip install -r requirements.txt
pip install -r requirements-dashboard.txt     # needed for the app and charts
python data/download.py                       # about 600 MB of market data""")

callout("You almost certainly do not need to train anything",
        "Every trained program is already saved in the project - 79 files, about 15 MB. Someone downloading "
        "this project fresh can reproduce every result in this guide <b>without training for a single "
        "second</b>.")

h2("Check everything works")
code("""python -m pytest tests -q                     # should say 214 passed, about 2 minutes
streamlit run app.py --server.fileWatcherType none""")

h2("Reproduce the results")
code("""python scripts/calibrate.py                   # measure how much the price moves
python watchdog/dataset.py --episodes 150     # record what the detective would see
python watchdog/evaluate_watchdog.py --tag cov    # the detective vs the old rule
python scripts/label_observability.py         # the ceiling - run this one
python watchdog/supervised_baselines.py train # is the fancy method needed? (no)
python watchdog/anonymised_experiment.py      # the fair test, answer sheet removed
python watchdog/posthoc_rescore.py            # every version on every dataset""")

h2("Train from scratch, if you really want to")
code("""python training/train_spoofer.py SPOOFER-04 --timesteps 600000
python watchdog/train_watchdog.py --timesteps 400000 --norm-obs --ent-coef 0.01""")
p("<b>Both of those flags on the second line are required.</b> Without "
  "<font face='Courier'>--norm-obs</font> the detective learns to never flag anything at all. Without the "
  "other one it stops exploring too early and gets stuck.", Small)

h2("A safety message you may see")
code("""$ python watchdog/drip_test.py
drip_test.py: refusing to overwrite results that already exist:
  results/drip_test.json
  results/drip_test.md
These record a run that was meant to happen once. Pass --rerun if you really
intend to replace them, or move them aside first.""")
p("Some tests are meant to be run <b>once only</b>, because we committed to the result in advance. The "
  "project now physically stops you from overwriting them by accident.", Small)

h2("What you need to run it")
table([
    ["Thing", "What is needed"],
    ["Machine", "An ordinary laptop. No graphics card is required - everything here runs on the processor."],
    ["To just look at the results", "Seconds. Open the app or the HTML file; nothing is computed."],
    ["To re-check all the results", "A few minutes per script. Nothing needs training."],
    ["To run the test suite", "About 2 to 3 minutes for all 214 checks."],
    ["To train a cheater from scratch", "Roughly 20 to 40 minutes for 600,000 practice steps."],
    ["To train a detector from scratch", "Roughly 1 to 2 hours. You should not need to - they are saved."],
    ["Disk space", "About 600 MB for the market data, plus 15 MB of saved programs."],
], [52 * mm, 112 * mm])

h2("Where to find things")
table([
    ["File", "What it is"],
    ["README.md", "The manual - every command and every result"],
    ["REPORT.md", "The formal write-up: method, results, discussion, limitations"],
    ["HANDOFF.md", "A friendly introduction for the next person to work on it"],
    ["docs/FINAL_AUDIT.md", "The audit: what was checked, what was broken, the verdict"],
    ["docs/FEATURE_DICTIONARY.md", "Every input number explained, and whether it leaks the answer"],
    ["docs/DEMO_GUIDE.md", "The minute-by-minute plan for a 10-minute presentation"],
    ["docs/MODEL_COMPARISON.md", "Every method compared fairly on the same data"],
    ["docs/EXPERIMENT_PROTOCOL.md", "The rules we set ourselves before running the tests"],
    ["docs/CHANGELOG.md", "What changed over time and why"],
], [54 * mm, 110 * mm])

# ───────────────────────────── 9. Q&A ─────────────────────────────
story.append(PageBreak())
h1("9. Who built what")

p("This was a two-person project, and the two halves were built separately before being merged into one "
  "repository. Both people independently found the same central flaw, from opposite directions - which is "
  "the strongest evidence that the flaw is real.", Lead)

table([
    ["Work", "Who"],
    ["The simulated market, the cheating programs, the first detector, the first two pre-registered tests, "
     "the price-impact measurement, the synthetic second market", "Shared foundation"],
    ["<b>The detector fix</b> - the causal label, the myopic objective, the varied-timing training, and "
     "the CP-7 detectors", "Arnav"],
    ["Two further pre-registered tests (DRIP and REVERSAL), the supervised comparisons, the "
     "label-definition rule, and the real-market rule check", "Arnav"],
    ["The audit - the train/test overlap bug, the zero-parameter signature finding, the test-without-"
     "attribution experiment, the per-instrument investigation", "The other half"],
    ["The interface, the documentation set, and the automated checks that keep the documents honest",
     "The other half"],
], [122 * mm, 42 * mm])

callout("The part worth mentioning out loud",
        "Arnav built <b>LABEL-RULE</b> - the definition of cheating written out as a rule - and recorded it "
        "as <i>\"the ceiling, perfect by construction, not a competitor\"</i>. Separately, the other half "
        "built the two-feature signature and measured 0.997. Same finding, two people, two routes. Neither "
        "was checking the other's work.")

p("The two lines of work had drifted apart in version control and were reconciled into one. Every commit "
  "from both is preserved in the project history, and the point where they split is tagged so it can "
  "always be recovered.", Small)

story.append(PageBreak())
h1("10. Questions you will be asked")

QA = [
 ("Can this catch real market cheating?",
  "No, and we never claim it can. The real data does not say <i>who</i> placed each order, and our detective "
  "needs that. So it cannot even be switched on against real market data - only the old simple rule can. "
  "There is also no public dataset anywhere with real cheating marked in it, so nobody could check the "
  "answer even if we ran it."),
 ("Your detector scores 0.98. Isn't that too good to be true?",
  "Yes - which is exactly why we checked. Simply comparing two of the numbers we were feeding it, with "
  "<b>no learning at all</b>, scores 0.997. We had reached a bar that needs no cleverness. We found that "
  "ourselves and put it on the front page rather than hiding it."),
 ("Then why is 'reinforcement learning' in the project title?",
  "For the detective, honestly, it should not be - a much simpler method matches it in minutes. "
  "Reinforcement learning earns its place on the <b>cheater</b> side, where there is no right answer to "
  "copy from and the program genuinely has to work the strategy out for itself. Our report says this "
  "plainly."),
 ("So what did the project actually achieve?",
  "Two things. First, programs taught themselves to cheat profitably, and we proved the money comes from "
  "the trickery and not from lucky trading. Second, and more useful to anyone else: we showed that a "
  "<b>labelling mistake</b> had quietly turned 'spot the cheating' into 'spot the cheater', and that fixing "
  "it took the score from 22 out of 100 to 98 out of 100."),
 ("How do I know you did not just tune the tests until they passed?",
  "Because we wrote down the test, the attacker and the expected result <b>before</b> creating any of that "
  "test's data, and saved that to version control with a timestamp. Anyone can check the order of events. "
  "Each test was run once. The first two failed, and we published that."),
 ("Was there a mistake in how you split training and testing data?",
  "Yes, and we found and fixed it. Our test data overlapped our training data - 33 of 50 test runs shared "
  "market data with training. We fixed it by giving each split a separate stretch of the day, and then "
  "<b>measured</b> how much the mistake had flattered us: 0.031, which is smaller than the normal variation "
  "between training runs."),
 ("Is your simulation realistic?",
  "Partly, and we list exactly where it is not. The price does react to our orders, using a model measured "
  "from real data. But historical traders cannot react to us, there is no queuing for position, no trading "
  "fees and no delays. Only one big order per side is allowed, which is why one cheating style had to be "
  "scripted by hand rather than learned."),
 ("How much do the results change between training runs?",
  "A lot in one place, and we show it rather than hiding it. The Microsoft cheater's profit ranges from "
  "+$3,628 to +$15,886 across three runs. Intel is steady. Where a method has no randomness, the variation "
  "is exactly zero - that is correct, not a bug."),
 ("What happens if you take away the numbers that give the answer?",
  "The score drops from 0.997 to <b>0.392</b>. That is the honest difficulty of the real problem. It still "
  "beats flagging everything, so there is genuine signal in public data - but it is modest, and we say so."),
 ("Why does that work for some companies and not others?",
  "We do not know, and we say so. It works on Intel and Microsoft and fails on Apple, Google and Amazon. We "
  "tested two explanations and <b>both were wrong</b>: the cheater behaves identically on all five, and the "
  "disturbance it causes is actually <i>smaller</i> on the two where we succeed. With only five companies we "
  "cannot prove a cause."),
 ("What is the biggest weakness?",
  "One day of data, five companies, and only two where cheating pays at all. That limits almost every "
  "conclusion here, and no amount of extra programming fixes it - it needs more data."),
 ("What would you do next?",
  "Stop telling the detective which orders belong to whom, and make it work it out from public market "
  "activity alone. That is the change that turns this into a genuinely hard and realistic problem. "
  "Everything else is secondary until that is done."),
 ("How do I know the numbers in your report match your code?",
  "The project checks itself. Fourteen automatic tests compare each headline number in the documents against "
  "the file that produced it - including the number of tests we claim to have. If a document goes out of "
  "date, the tests fail."),
 ("Who actually did which part?",
  "It was two people. The simulated market, the cheating programs and the first detector were the shared "
  "foundation. <b>Arnav</b> built the fix that made the detector work - the corrected labelling, the "
  "myopic objective and the varied-timing training - plus two of the four pre-registered tests and the "
  "supervised comparisons. The audit, the attribution-free experiment, the interface and the documentation "
  "came from the other half. Section 9 has the full split."),
 ("What do you need to run it?",
  "An ordinary laptop. No graphics card. Looking at the results takes seconds because every trained program "
  "is already saved in the project. Only if you insisted on training from scratch would you need an hour or "
  "two."),
 ("Has anyone else reproduced this?",
  "Yes. The results were produced on a Mac and re-run on a Windows machine. The detective's scores matched "
  "to three decimal places, and two complete test datasets were rebuilt from scratch and gave identical "
  "answers across 1,164 and 1,615 separate numbers."),
]
for i, (q, a) in enumerate(QA, 1):
    story.append(KeepTogether([
        Paragraph(f'<font color="{ACCENT.hexval()}"><b>Q{i}. {q}</b></font>', QStyle),
        Paragraph(a, st("a", parent=Body, leftIndent=6, spaceAfter=9))]))

story.append(PageBreak())
h1("The whole project in one paragraph")
p("We built a simulated stock market out of one real day of NASDAQ trading. Into it we put computer programs "
  "whose only instruction was to make money - and they taught themselves to cheat, by placing big fake "
  "orders to push the price around. We proved the money came from the trickery by switching that effect off "
  "and watching the same trading lose money. We then built a second program to catch them. At first it "
  "failed on cheating styles it had not seen. The cause turned out to be our own <b>labelling</b>, not the "
  "program: we had been marking the whole time a fake order sat there, which taught it to recognise "
  "<i>people</i> rather than <i>actions</i>. Fixing that took it from catching 22 in 100 to 98 in 100. But "
  "we then discovered that simply comparing two of the numbers we were feeding it - with no learning at all "
  "- scores even higher. So the detective half is an honest negative result, and we published it that way.",
  Lead)

gap(10)
callout("The fair summary",
        "This project is correct, repeatable and honest. Its weakness is not a mistake - it is a limit: one "
        "day of data. Its strength is that it found, measured and published the flaw in its own setup before "
        "anybody else could.")

doc.build(story)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
