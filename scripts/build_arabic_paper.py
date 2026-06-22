"""
Atheer Paper - Arabic Summary Version
=====================================
A concise Arabic version of the revised Atheer paper, generated as a PDF
using ReportLab with proper Arabic shaping via arabic-reshaper and python-bidi.
"""

import os
import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage, KeepTogether
)


# ============================================================
# Font registration - use FreeSerif which supports Arabic glyphs
# ============================================================
FONT_REGULAR = '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf'

pdfmetrics.registerFont(TTFont('ArabicSerif', FONT_REGULAR))
pdfmetrics.registerFont(TTFont('ArabicSerifBold', FONT_BOLD))


def ar(text: str) -> str:
    """Reshape Arabic text for proper RTL display in ReportLab."""
    if not text:
        return text
    return get_display(arabic_reshaper.reshape(text))


# ============================================================
# Styles
# ============================================================
PRIMARY = HexColor('#003B71')
ACCENT = HexColor('#C44536')
LIGHT_BG = HexColor('#F5F5F0')
TEXT_COLOR = HexColor('#1A1A1A')

styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    'ArabicTitle',
    parent=styles['Title'],
    fontName='ArabicSerifBold',
    fontSize=18,
    leading=24,
    alignment=TA_CENTER,
    textColor=PRIMARY,
    spaceAfter=10,
)

style_subtitle = ParagraphStyle(
    'ArabicSubtitle',
    fontName='ArabicSerif',
    fontSize=11,
    leading=15,
    alignment=TA_CENTER,
    textColor=TEXT_COLOR,
    spaceAfter=6,
)

style_authors = ParagraphStyle(
    'ArabicAuthors',
    fontName='ArabicSerifBold',
    fontSize=11,
    leading=15,
    alignment=TA_CENTER,
    textColor=TEXT_COLOR,
    spaceAfter=4,
)

style_h1 = ParagraphStyle(
    'ArabicH1',
    fontName='ArabicSerifBold',
    fontSize=14,
    leading=20,
    alignment=TA_RIGHT,
    textColor=PRIMARY,
    spaceBefore=14,
    spaceAfter=8,
)

style_h2 = ParagraphStyle(
    'ArabicH2',
    fontName='ArabicSerifBold',
    fontSize=12,
    leading=17,
    alignment=TA_RIGHT,
    textColor=ACCENT,
    spaceBefore=8,
    spaceAfter=4,
)

style_body = ParagraphStyle(
    'ArabicBody',
    fontName='ArabicSerif',
    fontSize=10.5,
    leading=17,
    alignment=TA_RIGHT,
    textColor=TEXT_COLOR,
    spaceAfter=8,
    wordWrap='RTL',
)

style_bullet = ParagraphStyle(
    'ArabicBullet',
    fontName='ArabicSerif',
    fontSize=10.5,
    leading=16,
    alignment=TA_RIGHT,
    rightIndent=15,
    textColor=TEXT_COLOR,
    spaceAfter=4,
    wordWrap='RTL',
)

style_caption = ParagraphStyle(
    'ArabicCaption',
    fontName='ArabicSerif',
    fontSize=9,
    leading=12,
    alignment=TA_CENTER,
    textColor=HexColor('#555555'),
    spaceAfter=8,
)


# ============================================================
# Build document
# ============================================================
def build_doc(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title='Atheer - Arabic Summary',
        author='Nabil Al-Mekhlafi, Ahmed Al-Mutawakel',
    )

    story = []

    # ---- Title block ----
    story.append(Paragraph(ar('معمارية مرنة لدفع الموبايل أوفلاين باستخدام NFC و HCE: نهج مُحسّن التكلفة للبيئات منخفضة البنية التحتية'), style_title))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(ar('نسخة عربية مختصرة من الورقة المنقّحة (v2.0)'), style_subtitle))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(ar('نبل المخلافي، أحمد المتوكل'), style_authors))
    story.append(Paragraph(ar('كلية علوم الحاسوب - جامعة صنعاء - اليمن'), style_subtitle))
    story.append(Spacer(1, 8 * mm))

    # ---- Abstract ----
    story.append(Paragraph(ar('الملخص'), style_h1))
    abstract = (
        'تُعاني أنظمة الدفع الإلكتروني في المناطق منخفضة البنية التحتية من انقطاع الاتصال '
        'المتكرر وتكاليف الأجهزة الباهظة. في اليمن، على سبيل المثال، لا يتمتع سوى 17.7% من '
        'السكان بالوصول إلى الإنترنت، وتتطلب خدمات المحافظ الحالية (الكرمي، جوالي، جيب، '
        'مفلوس) اتصالاً دائماً بالسحابة، مما يُجبر التجار والعملاء على العودة للنقد عند انقطاع '
        'الاتصال. نُقدّم في هذه الورقة "أثير"، وهي معمارية دفع أوفلاين-أولاً تجمع بين '
        'مُعالِج خلفي رباعي الطبقات، وSDK أندرويد باستخدام Host Card Emulation (HCE)، '
        'ومفاتيح ذات استخدام محدود (LUKs)، ونموذج جديد لتوجيه المعاملة عبر بيانات الهاتف '
        'المحمول مع فوترة شريكة مدعومة من المحفظة الشريكة. '
        'بدلاً من الاعتماد على Private APN كما في النسخة السابقة، يُوجّه أثير حركة التسوية '
        'عبر بيانات الهاتف المحمول القياسية بحمولة مُحسّنة بدقة (180 بايت)، بينما تتحمل '
        'المحفظة الشريكة تكلفة البيانات عبر عقد B2B. أظهرت محاكاة الأحداث المتقطعة (DES) '
        'عبر ستة مستويات حمل (5-500 TPS، N=10) أن مسار بيانات الهاتف يحقق نسبة نجاح 97.6% '
        'عند 500 TPS وزمن P95 = 672 مللي ثانية، بينما ينهار خط الأساس للإنترنت العام إلى '
        '76.2%. يُبيّن نموذج التكلفة أن نفقات البيانات تظل أقل من 0.5% من إيرادات MDR '
        'للمحفظة الشريكة، مما يجعل المعمارية مستدامة اقتصادياً للنشر الوطني في الاقتصادات الهشّة.'
    )
    story.append(Paragraph(ar(abstract), style_body))

    # ---- Keywords ----
    kw = 'الكلمات المفتاحية: الدفع الأوفلاين للموبايل، SoftPOS، NFC، HCE، Android SDK، توجيه بيانات الهاتف، الفوترة الشريكة المدعومة، تحسين الحمولة، الإدماج المالي.'
    story.append(Paragraph(ar(kw), style_body))

    # ---- Section 1 ----
    story.append(Paragraph(ar('1. المقدمة'), style_h1))
    sec1 = (
        'أسرع اعتماد الهواتف الذكية وتقنية NFC من تطور الاقتصادات غير النقدية، لكن هذا التطور '
        'لم يكن متكافئاً بين المناطق. فمع نسبة انتشار للإنترنت تبلغ 17.7% فقط [1]، عانت البنية '
        'التحتية للشبكة في اليمن من سنوات من الصيانة المؤجّلة، مما أدى إلى انقطاعات مزمنة '
        'وزمن استجابة عالٍ وعرض نطاق منخفض [2]. '
        'يتمتع العملاء اليمنيون بالوصول إلى أنظمة محافظ موبايل مقدّمة من رواد الإدماج المالي '
        'مثل الكرمي، جوالي، جيب، ومفلوس. لكن هذه الأنظمة التقليدية تعتمد على SMS أو USSD '
        'كقنوات بديلة عند انقطاع الاتصال، وهي قنوات بطيئة لا توفّر تشفيراً من طرف إلى طرف، '
        'مما يُقلّل بشكل كبير من تجربة المستخدم [3]. عند انقطاع الشبكة، لا يملك التجار '
        'والعملاء خياراً سوى العودة للنقد. '
        'للتعامل مع هذه القيود، قمنا ببناء "أثير"، وهو SDK أندرويد خاص ومُعالِج خلفي بمعمارية '
        'رباعي الطبقات. يقوم SDK من جانب العميل بتحميل عدة دفعات من الرموز المشفّرة '
        '(Limited Use Keys) ويستخدم NFC بالاقتران مع Host Card Emulation لإتمام المعاملات '
        'دون الحاجة إلى اتصال إنترنت نشط على جهاز العميل. ثم يُرسل جهاز التاجر الحمولة الموقّعة '
        'عبر بيانات الهاتف المحمول القياسية إلى مُبدّل بوابة أثير للتحقق وتحديث السجل.'
    )
    story.append(Paragraph(ar(sec1), style_body))

    # ---- Contributions ----
    story.append(Paragraph(ar('المساهمات الرئيسية:'), style_h2))
    contributions = [
        'بنية ثنائية المكوّنات (SDK حافة + مُبدّل بوابة) بمعمارية رباعي الطبقات تدمج توجيه بيانات الهاتف المحمول مع الفوترة الشريكة المدعومة، والتحقق متعدد المزوّدين، وSoftPOS بدون تكلفة رأسمالية.',
        'cryptogram بيومتري مُصرَّح به مسبقاً يربط مفتاح التوقيع بـ TEE عبر قيد بيومتري ونافذة زمنية مدتها 60 ثانية، مع تحقق Zero-Trust من جانب الخادم.',
        'نموذج تكلفة رسمي يُثبت أن حمولة مُحسّنة بدقة (180 بايت) تسمح للمحفظة الشريكة بتحمّل تكلفة بيانات التاجر مع إبقاء نفقات البيانات أقل من 0.5% من إيرادات MDR، مُتحقَّق منه عبر محاكاة DES على ستة مستويات حمل (5-500 TPS) تُظهر نجاحاً 97.6% عند ذروة الحمل.',
    ]
    for c in contributions:
        story.append(Paragraph('• ' + ar(c), style_bullet))

    # ---- Section 2 ----
    story.append(Paragraph(ar('2. الأعمال ذات الصلة'), style_h1))
    sec2 = (
        'أثير الاهتمام البحثي بالشبكات المالية التي تعمل أوفلاين. فبالنسبة لصندوق النقد الدولي، '
        'تُعدّ الوظيفة الأوفلاين شرطاً مسبقاً للإدماج المالي في الاقتصادات النامية [4]. كذلك '
        'استثمر البنك الدولي في تطوير البنية التحتية الرقمية لليمن [5]. لكن معظم النشرات '
        'المحلية تتجاهل الاحتكاكات التشفيرية والمعمارية الأساسية، مما يحدّ من قدرتها على التوسّع. '
        'تُعدّ قاعدة مستخدمي حلول SMS وUSSD في اليمن الأكبر حتى الآن، وتستفيد محافظ مثل '
        'الكرمي وجيب من قاعدة المستخدمين هذه لأن خدماتها لا تتطلب اتصالاً نشطاً بالإنترنت. '
        'لكن هذه القنوات تعاني من نقاط ضعف تشفيرية كبيرة، بما في ذلك غياب التشفير من طرف '
        'إلى طرف، مما يُعرّض الرسائل لهجمات man-in-the-middle والتنصّت [3]. '
        'من جهة أخرى، قدّمت الجهود المحلية الحديثة أنظمة NFC قائمة على الدوائر، لكنها تتطلّب '
        'من التجار شراء أجهزة POS خاصة [6]، وهو حاجز تكلفة رأسمالية لا يستطيع معظم '
        'الشركات الصغيرة والمتوسطة تحمّله. ظهر اتجاه بحثي حديث نحو النماذج البرمجية '
        '(SoftPOS) حيث تتحوّل أجهزة الموبايل العادية إلى أجهزة POS، حيث أظهر Oladapo et al. '
        'أن دمج HCE مع Limited Use Key tokenization يُحوّل الأجهزة المحمولة إلى أجهزة قبول '
        'مرنة مع معالجة هجمات إعادة التشغيل [7]. '
        'الفجوة البحثية: لا تجمع أي دراسة سابقة بين استقلالية العميل عن الإنترنت، وSoftPOS '
        'بدون تكلفة رأسمالية، ونموذج توجيه بيانات موبايل مُحسّن التكلفة في نهج واحد. هذا هو '
        'الفجوة التي يعالجها أثير.'
    )
    story.append(Paragraph(ar(sec2), style_body))

    # ---- Section 3 ----
    story.append(Paragraph(ar('3. المنهجية'), style_h1))
    sec3 = (
        'استخدمنا منهجية Design Science Research (DSR) [8, 9]، وهي نهج منهجي لتطوير وتقييم '
        'القطع التقنية (IT artifacts) التي تُعالج مشاكل العالم الحقيقي. باستخدام DSR، قمنا '
        'بتصميم وبناء وتقييم أثير. '
        'نظراً لأن الاختبار الحيّ على بنية تحتية مصرفية حقيقية في منطقة نزاع نشط غير عملي، '
        'اعتمدنا محاكاة الأحداث المتقطعة (DES) باستخدام SimPy. تقيس المحاكاة زمن الاستجابة '
        'P95، ونسب النجاح، والعبء الذي يتحمّله مُبدّل أثير تحت ظروف تشغيل واقعية. '
        'يتم توفير الشيفرة المصدرية للمحاكاة كقطعة بحثية مفتوحة المصدر [10].'
    )
    story.append(Paragraph(ar(sec3), style_body))

    # ---- Ethical considerations (new section addressing Reviewer 2) ----
    story.append(Paragraph(ar('3.1 الاعتبارات الأخلاقية'), style_h2))
    ethics = (
        'التزاماً بمبادئ DSR ولمعالجة المخاوف الأخلاقية المشروعة المُثارة في مراجعة سابقة، '
        'لا تُقيّم هذه الدراسة أو تختبر أمن أي مؤسسة مالية مُسمّاة في اليمن. الإشارات إلى '
        'المحافظ مثل الكرمي، جوالي، جيب، ومفلوس تظهر فقط في سياق وصف السوق؛ ولا تُقدّم أي '
        'ادعاءات حول وضعها التشفيري. المحاكاة تركيبية بالكامل، لا تستخدم أي بيانات مصرفية '
        'حقيقية، ولا تتطلب إذناً من أي مشغّل. جميع افتراضات التكلفة مستمدة من مراجع تسعير '
        'متاحة للعموم (ITU, GSMA, Cable.co.uk).'
    )
    story.append(Paragraph(ar(ethics), style_body))

    # ---- Section 4: Architecture ----
    story.append(Paragraph(ar('4. المعمارية المقترحة'), style_h1))
    sec4 = (
        'تجنّباً للنماذج الشاملة لكل البيئات، بنينا أثير بمعمارية موزّعة من أربع طبقات متميزة '
        '(الشكل 1). طبقة الحافة (Edge Layer) تضم تطبيق SoftPOS للتاجر وSDK العميل الأوفلاين. '
        'طبقة الشبكة (Network Layer) تُعالج التوجيه عبر بيانات الهاتف المحمول مع نموذج فوترة '
        'شريكة مدعومة. طبقة المُبدّل (Switch Layer) تعمل كبوابة معاملات مستقلة. الطبقة '
        'الأخيرة، طبقة التكامل (Integration Layer)، تتواصل مع وحدات HSM والسجل المصرفي الرئيسي.'
    )
    story.append(Paragraph(ar(sec4), style_body))

    # Add architecture figure
    arch_img_path = '/home/z/my-project/scripts/figures/fig1_architecture.png'
    if os.path.exists(arch_img_path):
        story.append(KeepTogether([
            RLImage(arch_img_path, width=15 * cm, height=8 * cm),
            Paragraph(ar('الشكل 1: المعمارية الرباعية المُحدّثة لأثير'), style_caption),
        ]))

    # ---- 4.1 Edge Layer ----
    story.append(Paragraph(ar('4.1 طبقة الحافة: المكتبة المعيارية القابلة للدمج'), style_h2))
    edge = (
        'مكتبة العميل هي وحدة أندرويد خفيفة الوزن تندمج في تطبيقات المحافظ الحالية دون إعادة '
        'هيكلة الشيفرة. يستخدم النظام خمسة مكوّنات لإجراء المعاملات الأوفلاين: '
        '(1) مدير المفاتيح التشفيرية الذي يحتفظ بمفتاحين في TEE؛ '
        '(2) خزنة الرموز الآمنة التي تخزّن LUKs بدلاً من PANs؛ '
        '(3) محرّك HCE الذي يعمل عند اكتشاف إشارة NFC؛ '
        '(4) وحدة توجيه الشبكة التي تراقب حالة الاتصال؛ '
        '(5) مخزن البيانات المحلي المشفّر.'
    )
    story.append(Paragraph(ar(edge), style_body))

    # ---- 4.2 Network Layer ----
    story.append(Paragraph(ar('4.2 طبقة الشبكة: توجيه بيانات الهاتف مع الفوترة الشريكة المدعومة'), style_h2))
    network = (
        'تحلّ هذه الطبقة محلّ نموذج Private APN الأصلي بنهج أكثر قابلية للنشر. بدلاً من '
        'تطلّب نفق خلوي مخصص (الذي يعتمد على اتفاقيات B2B مع مشغّلي الشبكات المحمولة يصعب '
        'الحصول عليها في الاقتصادات الهشّة)، يوجّه أثير حركة التسوية عبر اتصال بيانات الهاتف '
        'المحمول القياسي للتاجر. الرؤية الأساسية أن الحمولة الناتجة صغيرة جداً بحيث تصبح '
        'التكلفة ضئيلة، ويمكن للمحفظة الشريكة تحمّلها عبر عقد B2B.'
    )
    story.append(Paragraph(ar(network), style_body))

    # ---- 4.3 Payload Optimization ----
    story.append(Paragraph(ar('4.3 تحسين الحمولة: 180 بايت'), style_h2))
    payload = (
        'يتم ضغط حزمة التسوية بدقة إلى 180 بايت عن طريق تقليل أطوال الحقول وإزالة الترويسات '
        'المكرّرة. تتكوّن الحزمة من: ترويسة توجيه (32 بايت)، رمز LUK (32 بايت)، ATC '
        '(4 بايت)، حقل المبلغ (8 بايت)، nonce (16 بايت)، طابع زمني (8 بايت)، توقيع ECDSA '
        '(64 بايت)، وAES-GCM IV+tag (28 بايت). بعد ضغط الترويسة (ترميز متغيّر الطول '
        'لـ merchant ID وtransaction ID)، يتم تقليل الحجم الفعلي من 192 بايت إلى 180 بايت.'
    )
    story.append(Paragraph(ar(payload), style_body))

    # ---- 4.4 Partner-Subsidized Billing ----
    story.append(Paragraph(ar('4.4 نموذج الفوترة الشريكة المدعومة'), style_h2))
    billing = (
        'في هذا النموذج، توقّع المحفظة الشريكة اتفاقية B2B مع مشغّل الشبكة المحمولة للتاجر '
        'لتعويض تكاليف البيانات عن حركة أثير المُحدّدة. تُحدّد المحفظة حركتها عبر تثبيت '
        'شهادة TLS 1.3 وSNI مُحدّد؛ يُحاسب المشغّل هذه البايتات بشكل منفصل ويُفوتر المحفظة '
        'شهرياً. لأن الحمولة 180 بايت فقط لكل معاملة، فإن التكلفة تُصبح ضئيلة مقارنة بإيرادات '
        'MDR التي تربحها المحفظة على كل معاملة ناجحة (انظر القسم 7).'
    )
    story.append(Paragraph(ar(billing), style_body))

    # ---- Section 5: Workflow ----
    story.append(Paragraph(ar('5. سير عمل النظام'), style_h1))
    sec5 = (
        'يستخدم بروتوكول أثير التشغيلي نموذجاً هجيناً غير متزامن يُقسّم المعاملات إلى ثلاث '
        'مراحل: التزويد الأونلاين، تبادل NFC الأوفلاين الكامل، والتسوية النهائية عبر بيانات '
        'الهاتف المحمول.'
    )
    story.append(Paragraph(ar(sec5), style_body))

    story.append(Paragraph(ar('5.1 المرحلة 1: التهيئة التشفيرية وتزويد الرموز'), style_h2))
    p1 = (
        'تتطلّب هذه الخطوة اتصال إنترنت نشط للعميل لتحميل المواد التشفيرية. لمعالجة المخاوف '
        'حول الاعتماد على الشبكة العامة التي ينتقدها النظام، يدعم أثير مساري تزويد: '
        '(1) التزويد الداخلي عبر TLS 1.3: المسار الافتراضي حيث يقوم العميل بإنشاء قناة '
        'TLS 1.3 مع تثبيت الشهادة. بينما يستخدم هذا المسار الإنترنت العام، فإن الضمانات '
        'الأمنية تأتي من تشفير طبقة النقل وهوية الجهاز المُثبّتة بالأجهزة. '
        '(2) التزويد الخارجي في فروع البنك: للتجار عالي القيمة، يمكن تزويد الرموز في فرع '
        'بنك شريك عبر شبكة LAN خاصة.'
    )
    story.append(Paragraph(ar(p1), style_body))

    story.append(Paragraph(ar('5.2 المرحلة 2: التنفيذ الأوفلاين و Cryptogram البيومتري المُصرّح به مسبقاً'), style_h2))
    p2 = (
        'يتيح دور SDK كمقرّر محلي تصريحات الدفع الأوفلاين. يجب على المستخدم المصادقة أولاً '
        'على مستوى TEE. عند النجاح، يمكن للمستخدم توقيع "جلسة مسلّحة" لمدة 60 ثانية. خلال '
        'هذه النافذة، تُولّد المكتبة حمولة موقّعة تربط الرمز بتلك اللحظة الزمنية. عند اكتشاف '
        'NFC، يبدأ SoftPOS عملية المصافحة القياسية EMV. قبل الإرسال، تُجري المكتبة فحوصين: '
        '(1) صلاحية الرمز و(2) حالة الجلسة. بعد اجتياز الفحوصين، تُرسل المعاملة عبر APDU '
        'إلى جهاز التاجر، مما يُنهي الجلسة ويُعلّم الرمز كمستهلك.'
    )
    story.append(Paragraph(ar(p2), style_body))

    story.append(Paragraph(ar('5.3 المرحلة 3: توجيه البوابة والتسوية الآمنة عبر بيانات الهاتف'), style_h2))
    p3 = (
        'تعتمد التسوية على مسار داخلي يستخدم اتصال بيانات الهاتف المحمول للتاجر، مؤمّناً بـ '
        'TLS 1.3 ونموذج الفوترة الشريكة المدعومة. تُغلّف حزمة APDU ضمن حمولة Charge Request '
        'موحدة، تتكوّن من ترويسة توجيه عامة غير مشفّرة، وحمولة أساسية مشفّرة بـ AES، وتوقيع '
        'ECDSA للتاجر. تُنقل الحمولة عبر بيانات الهاتف القياسية (LTE/3G) مع TLS 1.3 وتثبيت '
        'الشهادة. نظراً لأن الحمولة 180 بايت فقط، فإنها تتّسع لمقطع TCP واحد وتكتمل في رحلة '
        'ذهاب وإياب واحدة تحت معظم ظروف الشبكة.'
    )
    story.append(Paragraph(ar(p3), style_body))

    # ---- Section 6: Security ----
    story.append(Paragraph(ar('6. تحليل الأمن'), style_h1))
    sec6 = (
        'يُدمج أثير الأمن عبر طبقات التخزين والحافة وتوجيه الشبكة. بدون عزل على مستوى الشبكة '
        '(كما يوفّره APN خاص)، يعوّض أثير عبر عدة آليات على مستوى التطبيق: '
        '(1) TLS 1.3 مع تثبيت الشهادة الذي يوفّر سرية forward secrecy ويُلغي هجمات التخفيض. '
        '(2) تقليل الحمولة إلى 180 بايت مما يُصغّر سطح تحليل الحركة. '
        '(3) تحديد معدّل الحافة وWAF في البوابة لتخفيف فيضان DDoS على مستوى التطبيق. '
        '(4) توجيه نقطة النهاية الوحيدة حيث يتواصل جهاز التاجر فقط مع بوابة أثير.'
    )
    story.append(Paragraph(ar(sec6), style_body))

    # ---- Threat model table ----
    threat_data = [
        [ar('التهديد'), ar('متجه الهجوم'), ar('التخفيف')],
        [ar('التنصّت'), ar('اعتراض NFC والشبكة'), ar('حمولة AES-256 GCM؛ تتابع صفري المعرفة للتاجر')],
        [ar('إعادة التشغيل'), ar('إعادة إرسال إشارة NFC'), ar('LUKs للاستخدام الواحد + ATC؛ سجلات nonce')],
        [ar('MitM'), ar('اعتراض الشبكة المفتوحة'), ar('TLS 1.3 + تثبيت شهادة ECDSA + توجيه نقطة واحدة')],
        [ar('العبث'), ar('تعديل المبلغ أو مُعرّف التاجر'), ar('توقيع ECDSA مُقيّد بيومترياً؛ HSM')],
        [ar('انتحال الهوية'), ar('جهاز بهوية مختلفة'), ar('البوابة تستخرج الهوية من السجل؛ تُتجاهل ادعاءات الجهاز')],
        [ar('سرقة الجهاز'), ar('السيطرة الفيزيائية'), ar('قفل TEE بيومتري؛ مفاتيح يتعذّر الوصول لها')],
        [ar('DDoS'), ar('فيضان البوابة'), ar('تحديد معدّل الحافة + WAF + تقليل حجم الحمولة')],
    ]
    threat_table = Table(threat_data, colWidths=[3.2 * cm, 5.2 * cm, 7.6 * cm])
    threat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONT', (0, 0), (-1, 0), 'ArabicSerifBold', 10),
        ('FONT', (0, 1), (-1, -1), 'ArabicSerif', 9),
        ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(threat_table)
    story.append(Paragraph(ar('الجدول 1: نموذج التهديد واستراتيجيات التخفيف (مُحدّث)'), style_caption))

    # ---- Section 7: Simulation ----
    story.append(Paragraph(ar('7. التقييم القائم على المحاكاة'), style_h1))
    sec7 = (
        'يركّز هذا القسم على جوانب الأداء والشبكة: زمن الاستجابة E2E، ونسبة نجاح المعاملة، '
        'وقابلية التوسّع. السؤال: إلى أي مدى يُحسّن توجيه بيانات الهاتف المحمول مع الفوترة '
        'الشريكة (S2) الإنتاجية والموثوقية مقارنةً بـ line الأساس للإنترنت العام (S1)؟'
    )
    story.append(Paragraph(ar(sec7), style_body))

    # ---- Results Table IV ----
    story.append(Paragraph(ar('7.1 النتائج'), style_h2))
    results_data = [
        [ar('TPS'), ar('نجاح S1 (%)'), ar('نجاح S2 (%)'), ar('P95 لـ S1 (ث)'), ar('P95 لـ S2 (ث)')],
        ['5', '99.50 ± 0.14', '98.32 ± 0.15', '1.272 ± 0.005', '0.459 ± 0.003'],
        ['25', '99.50 ± 0.06', '98.47 ± 0.05', '1.268 ± 0.003', '0.458 ± 0.001'],
        ['50', '99.14 ± 0.04', '98.48 ± 0.05', '2.144 ± 0.005', '0.458 ± 0.001'],
        ['100', '98.38 ± 0.03', '98.48 ± 0.04', '3.902 ± 0.011', '0.458 ± 0.001'],
        ['250', '96.02 ± 0.05', '98.17 ± 0.03', '9.164 ± 0.011', '0.538 ± 0.001'],
        ['500', '76.15 ± 0.05', '97.61 ± 0.02', '14.451 ± 0.004', '0.672 ± 0.000'],
    ]
    results_table = Table(results_data, colWidths=[1.5 * cm, 3.2 * cm, 3.2 * cm, 3.7 * cm, 3.7 * cm])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONT', (0, 0), (-1, 0), 'ArabicSerifBold', 9),
        ('FONT', (0, 1), (-1, -1), 'ArabicSerif', 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(results_table)
    story.append(Paragraph(ar('الجدول 2: ملخص أداء E2E (المتوسط ± 95% CI، N=10)'), style_caption))

    sec7_2 = (
        'يُوضّح الشكل 2 نسبة النجاح. يحافظ S2 (بيانات الهاتف المحمول) على نجاح تقريباً 97.6% '
        'عند 500 TPS، مع توسّع فعّال. يُظهر S1 (الإنترنت العام) انخفاضاً ثابتاً، حيث يصل إلى '
        '76.2% عند 500 TPS بسبب فقدان الحزمة المُجمّع وانتهاء المهلة. '
        'كما هو موضّح في الشكل 3، يُظهر S2 زمن P95 مستقراً قرب 672 مللي ثانية عند 500 TPS. '
        'بينما يختبر S1 تدهوراً ملحوظاً، حيث يصل زمن P95 إلى ما يقارب 14.5 ثانية عند 500 TPS، '
        'مع تجاوز غالبية المعاملات الفاشلة لمهلة E2E البالغة 15 ثانية.'
    )
    story.append(Paragraph(ar(sec7_2), style_body))

    # Add results figures
    for img_name, caption in [
        ('fig6_success_rate.png', 'الشكل 2: نسبة نجاح المعاملة. يحافظ S2 على ~97.6% عند 500 TPS'),
        ('fig7_p95_latency.png', 'الشكل 3: زمن استجابة P95. يبقى S2 مستقراً قرب 672 مللي ثانية'),
    ]:
        img_path = f'/home/z/my-project/scripts/figures/{img_name}'
        if os.path.exists(img_path):
            story.append(KeepTogether([
                RLImage(img_path, width=14 * cm, height=8 * cm),
                Paragraph(ar(caption), style_caption),
            ]))

    # ---- Section 8: Economic ----
    story.append(Paragraph(ar('8. التحليل الاقتصادي ونموذج الاستدامة'), style_h1))
    sec8 = (
        'الرؤية الاقتصادية الأساسية للمعمارية المُحدّثة هي أن حمولة مُحسّنة بدقة (180 بايت) '
        'تجعل توجيه بيانات الهاتف المحمول مستداماً اقتصادياً للمحفظة الشريكة. '
        'نُقسم هذا رسمياً عبر نموذج التكلفة التالي:'
    )
    story.append(Paragraph(ar(sec8), style_body))

    story.append(Paragraph(ar('8.1 نموذج التكلفة'), style_h2))
    cost_eq = (
        'تكلفة البيانات اليومية = (عدد المعاملات اليومية × حجم الحمولة) ÷ 1,048,576 × سعر البيانات لكل MB '
        'إيراد MDR اليومي = عدد المعاملات اليومية × متوسط المبلغ × معدّل MDR '
        'نسبة التكلفة = (تكلفة البيانات اليومية ÷ إيراد MDR اليومي) × 100%'
    )
    story.append(Paragraph(ar(cost_eq), style_body))

    story.append(Paragraph(ar('8.2 مثال عددي'), style_h2))
    example = (
        'باستخدام مُعاملات واقعية للسوق اليمني: 100,000 معاملة يومية (محفظة شريكة وطنية)، '
        'حمولة 180 بايت، سعر بيانات 1.00$ لكل MB (الحد الأعلى لـ Cable.co.uk لليمن)، '
        'متوسط مبلغ 5$ (معاملة صغيرة)، معدّل MDR = 1%: '
        'التكلفة اليومية = (100,000 × 180 ÷ 1,048,576) × 1.00$ ≈ 17.17$ '
        'الإيراد اليومي = 100,000 × 5$ × 0.01 = 5,000$ '
        'نسبة التكلفة = (17.17 ÷ 5,000) × 100% ≈ 0.34% '
        'تستهلك تكلفة البيانات تقريباً 0.34% من إيراد MDR، وهو أقل بكثير من عتبة 1% التي '
        'تهدّد الاستدامة الاقتصادية.'
    )
    story.append(Paragraph(ar(example), style_body))

    # Add cost figure
    cost_img = '/home/z/my-project/scripts/figures/fig8_cost_model.png'
    if os.path.exists(cost_img):
        story.append(KeepTogether([
            RLImage(cost_img, width=14 * cm, height=8 * cm),
            Paragraph(ar('الشكل 4: حساسية التكلفة: تكلفة البيانات كنسبة من إيراد MDR'), style_caption),
        ]))

    # ---- Section 9: Conclusion ----
    story.append(Paragraph(ar('9. الخاتمة'), style_h1))
    sec9 = (
        'فيما يتعلّق بأهداف التنمية الاقتصادية المستدامة، يُعدّ نظام أثير أحد حلول FinTech '
        'المبتكرة المُطوّرة خصيصاً للاقتصادات الهشّة وغير المطوّرة. لقد أظهرنا أن دمج NFC '
        'المبني على HCE مع توجيه بيانات الهاتف المحمول المُحسّن التكلفة يُنشئ أنظمة عملية '
        'وموثوقة، مُحقّقاً موثوقية 97.6% عند 500 TPS في ظروف الإجهاد المُحاكاة. '
        'يتصدّى النظام لثلاث مجالات مهمة: '
        '(1) الموثوقية وقابلية التوسّع: حيث أظهر مسار بيانات الهاتف المحمول نجاحاً 97.6% عند '
        '500 TPS، مُوفّراً بديلاً موثوقاً لأنظمة الإنترنت العام دون الحاجة إلى اتفاقيات APN '
        'خاصة مع المشغّلات. '
        '(2) الحماية والسرّية: حيث يُدمج النظام cryptogram بيومتري ونموذج Zero-Trust لضمان '
        'سلامة البيانات مع معالجة الانتحال والاحتيال. '
        '(3) الأثر الاقتصادي والإدماج المالي: من خلال ابتكار SoftPOS بدون تكلفة رأسمالية '
        'واتصال بيانات مدعوم من الشريك، يُقدّم النظام مزايا اقتصادية اجتماعية كبيرة. أثير '
        'هو اقتراح تقني وأيضاً مخطط عملي يبني جسراً من النظرية إلى الممارسة لخدمات المحافظ '
        'الإلكترونية والبنوك في المناطق المتأثرة بالنزاعات.'
    )
    story.append(Paragraph(ar(sec9), style_body))

    # ---- Section 10: Future Work ----
    story.append(Paragraph(ar('10. العمل المستقبلي'), style_h1))
    sec10 = (
        'تركّز أهدافنا قصيرة المدى على توسيع التعاون القطاعي لتطوير مشاريع تجريبية إضافية. '
        'نخطط لإجراء اختبارات ميدانية تتضمّن تعاونًا بين الجامعات والمحافظ الشريكة وشراكات '
        'قطاع عام-خاص. كما سنعمل على تأسيس شراكات مع مؤسسات مصرفية وهيئات تنظيمية لإنشاء '
        'مقترحات قابلة للتخصيص لتبنّي أطر التصريح الأوفلاين. '
        'أخيراً، ستخضع سمات الأمن المُؤسّسة في هذه الورقة لتحقق أمني رسمي باستخدام Tamarin '
        'Prover، مما يُنتج إثباتات مُتحقَّق منها آلياً لمقاومة إعادة التشغيل وصحة المصادقة '
        'تحت نموذج تهديد Dolev-Yao، مما ينقل الادعاءات الأمنية من حجج قائمة على المعمارية '
        'إلى حجج قائمة على التحقق الرسمي.'
    )
    story.append(Paragraph(ar(sec10), style_body))

    # ---- References (key) ----
    story.append(Paragraph(ar('المراجع الرئيسية'), style_h1))
    refs = [
        '[1] DataReportal, "Digital 2024: Yemen," Kepios, 2024.',
        '[2] Ookla, "Speedtest Global Index: Yemen," 2024.',
        '[3] GSMA, "The State of Mobile Internet Connectivity Report 2023."',
        '[4] IMF, "Fintech in the Middle East and Central Asia," Policy Paper 2022/004.',
        '[5] World Bank, "Yemen Digital Economy Assessment," 2023.',
        '[6] BrandPOS, "SoftPOS vs Traditional POS: A Complete Comparison," 2024.',
        '[7] Oladapo et al., "A Review of Tokenization and HCE in Mobile Payments," J. Cybersecurity Priv., 2024.',
        '[8] Hevner et al., "Design Science in IS Research," MIS Quart., 2004.',
        '[9] Peffers et al., "A Design Science Research Methodology," JMIS, 2007.',
        '[10] Al-Mutawakel, "Atheer Simulation Evaluation Artifact," GitHub, 2024.',
        '[11] Rescorla, "TLS 1.3," RFC 8446, IETF, 2018.',
        '[12] Cable.co.uk, "Worldwide Mobile Data Pricing 2024."',
    ]
    for ref in refs:
        story.append(Paragraph(ar(ref), ParagraphStyle(
            'ArabicRef', fontName='ArabicSerif', fontSize=9, leading=12,
            alignment=TA_RIGHT, textColor=TEXT_COLOR, spaceAfter=3, wordWrap='RTL'
        )))

    doc.build(story)
    print(f"Arabic paper generated: {output_path}")


if __name__ == "__main__":
    output = "/home/z/my-project/download/Atheer_Paper_v2_AR.pdf"
    build_doc(output)
    # Verify
    import subprocess
    result = subprocess.run(['pdfinfo', output], capture_output=True, text=True)
    print(result.stdout)
