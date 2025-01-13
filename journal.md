IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 23, NO. 9, SEPTEMBER 20151
Efficient Synthesis of Room Acoustics
via Scattering Delay Networks
Enzo De Sena,Member, IEEE, H
̈
useyin Hacıhabibo
̆
glu,Senior Member, IEEE,
Zoran Cvetkovi
́
c,Senior Member, IEEE, Julius O. Smith,Member, IEEE
Abstract—An  acoustic  reverberator  consisting  of  a  network
of  delay  lines  connected  via  scattering  junctions  is  proposed.
All  parameters  of  the  reverberator  are  derived  from  physical
properties  of  the  enclosure  it  simulates.  It  allows  for  simulation
of unequal and frequency-dependent wall absorption, as well as
directional  sources  and  microphones.  The  reverberator  renders
the  first-order  reflections  exactly,  while  making  progressively
coarser  approximations  of  higher-order  reflections.  The  rate  of
energy  decay  is  close  to  that  obtained  with  the  image  method
(IM)  and  consistent  with  the  predictions  of  Sabine  and  Eyring
equations.  The  time  evolution  of  the  normalized  echo  density,
which was previously shown to be correlated with the perceived
texture  of  reverberation,  is  also  close  to  that  of  IM.  However,
its  computational  complexity  is  one  to  two  orders  of  magnitude
lower,  comparable  to  the  computational  complexity  of  a  feed-
back  delay  network  (FDN),  and  its  memory  requirements  are
negligible.
Index  Terms—Room   acoustics,   acoustic   simulation,   digital
waveguide  network,  reverberation  time,  echo  density.
I.  INTRODUCTION
A
comprehensive account of the first fifty years of artificial
reverberation  in  [1]  identifies  three  main  classes  of
digital  reverberators:  delay  networks,  physical  room  models
Copyright
c
©2015  IEEE.  Personal  use  of  this  material  is  permitted.
However,  permission  to  use  this  material  for  any  other  purposes  must  be
obtained from the IEEE by sending a request to pubs-permissions@ieee.org.
http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7113826
Manuscript  received  February  19,  2015;  accepted  May  16,  2015.  Date  of
publication  June  01,  2015.  The  work  reported  in  this  paper  was  partially
funded by (i) EPSRC Grant EP/F001142/1, (ii) European Commission under
Grant  Agreement  no.  316969  within  the  FP7-PEOPLE  Marie  Curie  Initial
Training Network “Dereverberation and Reverberation of Audio, Music, and
Speech (DREAMS)”, and (iii) TUBITAK Grant 113E513 “Spatial Audio Re-
production Using Analysis-based Synthesis Methods”. The method presented
in this paper is protected by USPTO Patent n. 8908875.The associate editor
coordinating  the  review  of  this  manuscript  and  approving  it  for  publication
was Prof. Bozena Kostek.
Enzo De Sena is with ESAT–STADIUS, KU Leuven, Kasteelpark Arenberg
10,  3001  Leuven,  Belgium  (e-mail:  enzo.desena@esat.kuleuven.be).  This
work was done in part while he was with CTR, King’s College London.
Zoran  Cvetkovi
́
c  is  with  the  Centre  for  Telecommunications  Research
(CTR), King’s College London, Strand, London, WC2R 2LS, United Kingdom
(e-mail: zoran.cvetkovic@kcl.ac.uk).
H
̈
useyin Hacıhabibo
̆
glu is with Department of Modeling and Simulation, In-
formatics Institute, Middle East Technical University, Ankara, 06800, Turkey
(e-mail:  hhuseyin@metu.edu.tr).  This  work  was  done  in  part  while  he  was
with King’s College London.
Julius  O.  Smith  III  is  with  CCRMA,  Stanford  University,  Stanford,  CA
94304, USA.
This    paper    has    supplementary    downloadable    material    available    at
http://ieeexplore.ieee.org, provided by the author. The material includes sev-
eral  audio  samples  generated  using  the  proposed  room  acoustic  simulator.
Contact enzo.desena@esat.kulueven.be for further questions about this work.
and convolution-based algorithms. The earliest class consisted
of delay networks, which were the only artificial reverberators
feasible with the integrated circuits of the time. The first delay
network reverberator, as introduced by Schroeder, was a cas-
cade of several allpass filters, a parallel bank of feedback comb
filters and a mixing matrix [1]–[3]. Since then, a large number
of delay networks have been proposed and used commercially.
Most of these networks were designed heuristically and by trial
and error. Feedback delay networks (FDNs) were developed on
a more solid scientific grounding as a multichannel extension
of the Schroeder reverberator [4], [5], and consist of parallel
delay  lines  connected  recursively  through  a  unitary  feedback
matrix. The state-of-the-art FDN is due to Jot and Chaigne [6],
who  proposed  using  delay  lines  connected  in  series  with
multiband  absorptive  filters  to  obtain  a  frequency-dependent
reverberation time. FDN reverberators are still among the most
commonly  used  artificial  reverberators  owing  to  their  simple
design,  extremely  low  computational  complexity  and  high
reverberation  quality.  While  no  new  standard  seems  to  have
emerged yet, a number of more intricate networks have been
proposed more recently that show improvements over FDNs,
sometimes even in terms of computational complexity [7]–[9].
FDNs  are  structurally  equivalent  to  a  particular  case  of
digital waveguide networks (DWNs) [10], reverberators based
on the concept of digital waveguides, introduced in [11]. FDNs
and DWNs can also be viewed both as networks of multiport
elements, as explained by Koontz in [12]. DWNs consist of a
closed  network  of  bidirectional  delay  lines  interconnected  at
lossless junctions. DWNs have an exact physical interpretation
as  a  network  of  interconnected  acoustical  tubes  and  have  a
number  of  appealing  properties  in  terms  of  computational
efficiency  and  numerical  stability.  Reverberators  based  on
delay  networks  have  been  widely  used  for  artistic  purposes
in  music  production.  High-level  interfaces  enable  artists  and
sound  engineers  to  adjust  the  available  free  parameters  until
the intended qualities of reverberated sound are achieved.
In  the  domain  of  predictive  architectural  modeling,  virtual
reality and computer games, on the other hand, the objective of
artificial reverberators is to emulate the response of a physical
room given a set of physically relevant parameters [13], [14].
These  parameters  include,  for  instance,  the  room  geometry,
absorption  and  diffusion  characteristics  of  walls  and  objects,
and position and directivity pattern of source and microphone.
Various  physical  models  have  been  proposed  in  the  past
for  the  purpose  of  room  acoustic  synthesis.  Widely  used
geometric-acoustic  models  make  the  simplifying  assumption
Copyright
c
©2015 IEEE
arXiv:1502.05751v2  [cs.SD]  9 Jul 2015
2IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 23, NO. 9, SEPTEMBER 2015
that sound waves propagate as rays. In particular, the ray trac-
ing approach explicitly tracks the rays emitted by the acoustic
source  as  they  bounce  off  the  surfaces  of  a  modeled  enclo-
sure.  The  image  method  (IM)  is  an  alternative  algorithmic
implementation that formally replaces the physical boundaries
surrounding  the  source  with  an  equivalent  infinite  lattice  of
image  sources  [15],  [16].  Allen  and  Berkley  proved  that,  in
the  case  of  rectangular  rooms,  this  approach  is  equivalent  to
solving the wave equation provided that the walls are perfectly
rigid [15]. When the walls are not rigid, the results of the IM
are no longer physically accurate, but the method still retains
its geometric-acoustic interpretation. The IM can also be used
to  model  the  acoustics  in  arbitrary  polyhedra,  as  described
by Borish in [16]. The main advantage of geometric-acoustic
models in comparison to other physical models is their lower
–   although   still   considerable   –   computational   complexity.
However,  they  do  not  model  important  wave  phenomena
such  as  diffraction  and  interference.  These  phenomena  are
inherently  modeled  by  methods  based  on  time  and  space
discretization  of  the  wave  equation,  such  as  finite-difference
time-domain  (FDTD)  and  digital  waveguide  mesh  (DWM)
models [14], [17]–[19]. The main limitation of physical room
models consists of their significant computational and memory
requirements. While this may not be problematic for predictive
acoustic modeling applications, it is a significant limitation for
interactive applications, such as virtual reality (VR). Similarly,
convolutional  methods,  which  operate  by  filtering  anechoic
audio samples with measured room impulse responses (RIRs),
do not allow interactive operation unless interpolation among
an extensive number of RIRs is supported.
Virtual  reality  has  become  a  widespread  technology,  with
applications  in  military  training,  immersive  virtual  musical
instruments, archaeological acoustics, and, in particular, com-
puter  games.  Along  with  realistic  graphics  rendering,  spatial
audio  is  one  of  the  most  important  factors  that  affect  how
convincing   its   users   perceive   a   virtual   environment   [20].
The  aural  and  visual  components  should  be  consistent  and
mutually  supportive  so  as  to  minimise  cross-modal  sensory
conflicts [20].
Room acoustic synthesis for VR also requires flexibility in
terms  of  audio  output  devices.  For  example,  a  full  scale  VR
suite such as a CAVE automatic virtual environment (CAVE)
can use Ambisonics [21] or wave field synthesis (WFS), which
requires from tens to hundreds of loudspeaker channels [22].
In contrast, a portable game console has a two-channel output
that  may  be  used,  for  instance,  to  reproduce  binaural  audio
over headphones [23]. Typical home users, on the other hand,
commonly  use  stereophonic,  5.1  or  7.1  setups,  whereas  the
ultra  high  definition  TV  (UHDTV)  standard,  also  aimed  at
home users, makes provisions for 22.2 setups [24].
In  summary,  room  acoustic  synthesizers  for  VR  require
(i)  explicit  (tuning-free)  modeling  of  a  given  virtual  space,
(ii) scalability in terms of playback configuration, and (iii) low
computational  complexity.  On  the  other  hand,  of  the  three
classes  of  digital  reverberators,  (i)  convolutional  methods  do
not  allow  interactive  operation  without  extensive  tabulation
and  interpolation,  (ii)  delay  network  methods  do  not  model
explicitly a given virtual space, and (iii) physical models have
a high computational cost.
In  order  to  combine  the  appealing  properties  of  delay
networks and physical models, one possible approach consists
of  designing  delay  networks  that  have  parameters  with  an
explicit physical interpretation. Studies in [10], [25] and [26]
follow this direction. In [10] and [25] the length of the delay
lines of FDNs are chosen such that the lowest eigenfrequencies
of the room are reconstructed exactly. In [26], Karjalainenet
al.use a DWN with few junctions to model a rectangular room.
The rationale behind the design is to aggressively prune-down
a  DWM  in  order  to  reduce  the  complexity  while  retaining
an  acceptable  perceptual  result.  This  approach  has  various
advantages  but  requires  careful  manual  tuning  in  order  to
provide satisfactory results [26].
In  [27]  and  [28],  following  the  same  concept  of  DWN
structures as studied by Karjalainenet al.in [26], we presented
an architecture that has a number of appealing properties. The
proposed  architecture,  which  we  refer  to  as  scattering  delay
network  (SDN),  renders  the  direct-path  component  and  first-
order early reflections accurately both in time and amplitude,
while  producing  a  progressively  coarser  approximation  of
higher-order  reflections.  For  this  reason,  SDN  can  be  inter-
preted  as  an  approximation  to  geometric  acoustics  models
[15], [16]. SDNs thus approach the accuracy of full-scale room
simulation while maintaining computational efficiency on par
with  typical  delay  network-based  methods.  Furthermore,  the
parameters of SDN are inherited directly from room geometry
and  absorption  properties  of  wall  materials,  and  therefore  do
not require ad hoc tuning.
This  paper  further  explores  and  completes  the  design  pre-
sented in [27] and [28]. All design choices are now explained
on a physical basis. Furthermore, the paper includes a theoreti-
cal analysis of optimal scattering matrices, a comparison with
the  IM  in  terms  of  reverberation  time  and  normalized  echo
density [29], an analysis of the computational complexity and
of  the  memory  requirements,  and  an  analysis  of  the  modal
density  [30].  The  paper  is  organized  as  follows.  Section  II
presents  a  brief  overview  of  FDNs,  DWNs  and  models  pro-
posed  by  Karjalainenet  al.[26].  Section  III  describes  the
proposed SDN method. The properties of SDNs are studied in
Section  IV.  Section  V  presents  numerical  evaluation  results.
Section VI concludes the paper.
II.  BACKGROUND
The  proposed  SDN  reverberator  draws  inspiration  from
DWN   and   DWM   structures,   however   it   is   in   essence   a
recursive  linear  time-invariant  system.  Hence,  to  provide  a
comprehensive context, in this section we briefly review FDNs,
which  are  the  most  commonly  used  recursive  linear  time-
invariant  reverberators,  followed  by  a  more  detailed  review
of relevant DWN and DWM material.
A.  Feedback Delay Networks
The  canonical  feedback  delay  network  (FDN)  form,  as
proposed   by   Jot   and   Chaigne   [6],   is   shown   in   Fig.   1.
Here,bandcare   input   and   output   gains,   respectively,
D(z) =diag(z
−m
1
, z
−m
2
, ..., z
−m
N
)are  integer  delays,
DE SENAet al.:  EFFICIENT SYNTHESIS OF ROOM ACOUSTICS VIA SCATTERING DELAY NETWORKS3
Fig. 1.    Block diagram of the modified FDN reverberator as proposed by Jot
and Chaigne [6].
Fig. 2.Operation of a DWN around a junction withK= 4waveguides.
H(z) =diag(H
1
(z),..., H
N
(z))are absorption filters,T(z)
is  the  tone  correction  filter,gis  the  gain  of  the  direct  path,
andAis  the  feedback  matrix.  The  absorption  filters  can
be  designed  so  as  to  obtain  a  desired  reverberation  time  in
different  frequency  bands  [6],  or  to  match  those  calculated
from  a  measured  RIR  [31].  To  achieve  a  high-quality  late
reverberation, the feedback loop should be lossless, i.e. energy-
preserving, hence typically the feedback matrixAis designed
to be unitary. Each particular choice of the feedback matrix has
corresponding implications on subjective or objective qualities
of  the  reverberator  [32];  e.g.  in  the  particular  case  of  the
identity  matrix,A=I,  the  FDN  structure  reduces  toN
comb  filters  connected  in  parallel  and  acts  as  the  Schroeder
reverberator [3]. Note, however that unitary matrices are only
a subset of possible lossless feedback matrices [10], [33]; we
elaborate on this point in the next subsection.
B.  Digital Waveguide Networks
DWNs   consist   of   a   closed   network   of   digital   waveg-
uides  [11].  A  digital  waveguide  is  made  up  of  a  pair  of
delay  lines,  which  implement  the  digital  equivalent  of  the
d’Alembert solution of the wave equation in a one-dimensional
medium.  The  digital  waveguides  are  interconnected  at  junc-
tions, characterised by corresponding scattering matrices. Fig.
2  shows  an  example  of  four  digital  waveguides  with  length
D
1
,...,D
4
samples  that  meet  at  a  junction  with  scattering
matrixA.  In  general,  a  junction  scatters  incoming  wave
variablesp
+
=
[
p
+
1
,...,p
+
K
]
T
to  produce  outgoing  wave
variablesp
−
=
[
p
−
1
,...,p
−
K
]
T
according  top
−
=Ap
+
.
Note that if all digital waveguides are terminated by an ideal
non-inverting reflection, the DWN is structurally equivalent to
the  feedback  loop  of  an  FDN  with  feedback  matrixAand
delay-line lengths of2D
1
,...,2D
4
samples [10].
DWN  junctions  are  lossless.  In  this  context,losslessness
is  defined  according  to  classical  network  theory  [34].  In
particular,  a  junction  with  scattering  matrixAis  said  to  be
lossless if the input and outputtotal complex powerare equal:
p
+∗
Yp
+
=p
−∗
Yp
−
⇒A
∗
YA=Y(1)
whereYis  a  Hermitian  positive-definite  matrix  [10]  and
(·)
∗
denotes  the  conjugate  transpose.  The  quantityp
±∗
Yp
±
is  the  square  of  theelliptic  normofp
±
induced  byY.  It
can  be  shown  that  a  matrixAis  lossless  if  and  only  if  its
eigenvalues  lie  on  the  unit  circle  and  it  admits  a  basis  of
linearly independent vectors [10]. A consequence of this result
is that lossless feedback matrices can be fully parametrized as
A=T
−1
ΛT, whereTis any invertible matrix andΛis any
unit-modulus diagonal matrix [10].
The DWN can also be interpreted as a physical model for a
network of acoustic tubes. In this caseAassumes a particular
form. If we denote byy
i
the characteristic admittance of the
i-th  tube  and  byv
i
the  volume  velocity  of  thei-th  tube  at
the  junction,  the  continuity  of  pressure  and  conservation  of
velocity at the junction give, from [34]:
p
1
=p
2
=···=p
K
=p,(2)
v
1
+v
2
+···+v
K
= 0,(3)
respectively, wherep
i
=p
+
i
+p
−
i
denotes the acoustic pressure
of thei-th tube. Equations (2) and (3) imply that the pressure
pat the junction is given by
p=
2
∑
K
i=1
y
i
K
∑
i=1
y
i
p
+
i
=
2
∑
K
i=1
y
i
K
∑
i=1
y
i
p
−
i
,(4)
where  we  usedv
i
=v
+
i
+v
−
i
and  Ohm’s  law  for  traveling
wavesv
+
i
=y
i
p
+
i
andv
−
i
=−y
i
p
−
i
[34]. Sincep
−
i
=p−p
+
i
,
the scattering matrix can be expressed as
A=
2
〈1,y〉
1y
T
−I,(5)
where1=  [1,...,1]
T
,y=  [y
1
,...,y
K
]
T
,〈·,·〉denotes
the  scalar  product,  andIis  the  identity  matrix.  Observe
that  the  scattering  matrix  in  (5)  satisfies  equation  (1)  with
Y=diag{y
1
,...,y
K
}and  is  therefore  lossless.  In  this
physically-based  case,  the  square  of  the  elliptic  norm  ofp
±
induced byYhas the meaning of incoming/outgoing acoustic
power:p
±∗
Yp
±
=
∑
K
i=1
y
i
|p
±
i
|
2
=±
∑
K
i=1
v
∗
i
p
±
i
[10].
An  equivalent  formulation  of  DWNs  involves  normalized
pressure   waves,   defined   as ̃p
±
i
=p
±
i
√
y
i
.   In   this   case,
the  propagating  wave  variables ̃p
±
i
represent  the  square  root
of   the   traveling   signal   power   [10].   If   we   define
̃
Y=
diag(
√
y
1
,...,
√
y
K
),  the  normalized  output  wave  can  be
written  as
̃
p
−
=
̃
Yp
−
=
̃
YAp
+
=
̃
YA
̃
Y
−1
̃
p
+
=
̃
A
̃
p
+
,
where
̃
A=
̃
YA
̃
Y
−1
. The equivalent scattering matrix
̃
Acan
be expressed as
̃
A=
2
‖
̃
y‖
2
̃
y
̃
y
T
−I,(6)
which  is  a  Householder  reflection  around  the  vector
̃
y=
[
√
y
1
,...,
√
y
K
]
T
.  Such  Householder  matrices  will  be  also
used  in  the  context  of  SDN  reverberators,  proposed  in  the
next section, where they will exhibit some sought-after prop-
erties,  including  low  computational  complexity  and  desirable
normalised echo density profiles.
4IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 23, NO. 9, SEPTEMBER 2015
In order to inject energy in a DWN, various methods have
been  used  in  the  past,  ranging  from  attaching  an  additional
waveguide where the outgoing wave is ignored [11] or using
an  adapted  impedance  such  that  there  is  no  energy  reflected
along the outgoing wave, to more complex approaches [35]. A
common approach is to apply an external ideal volume velocity
source to the junction [19]. This is equivalent to superimposing
source  pressure,p
S
,  to  the  pressure  due  to  the  waveguides
meeting  at  the  junction,p,  thus  making  the  total  pressure  at
the junction equal to:
p=p
S
+p.(7)
In  the  context  of  FDTD  models,  a  source  that  injects  energy
in  this  way  is  called  asoft-source,  as  opposed  to  ahard-
source, which actively interferes with the propagating pressure
field  [36],  [37].  In  order  to  implement  equation  (7)  in  the
DWN  structure,  the  input  from  the  source  can  be  distributed
uniformly to incoming wave variables according to
p
+
=p
+
+
p
S
2
1.(8)
which  provides  the  intended  node  pressure  [38].  This  is  the
approach that will be used for injecting source energy in the
proposed SDN structures.
C.  Digital Waveguide Meshes
DWNs formed of fine grids of scattering junctions, referred
to  asdigital  waveguide  meshes,  are  used  to  model  wave
propagation in an acoustic medium [39]. Each spatial sample
in  a  digital  waveguide  mesh  (DWM)  is  represented  by  aK-
port scattering junction, connected to its geometric neighbors
over bidirectional unit-sample delay lines. In the typical case
of an isotropic medium,yis a constant vector, whileAand
̃
Aare identical and given by
A=
2
K
11
T
−I.(9)
We  will  refer  to  such  a  scattering  matrix  as  theisotropic
scattering matrix.
In the one-dimensional band-limited case, the DWM model
provides the exact solution of the wave equation [34]. In the
two [40] and three [41] dimensional cases, sound propagates in
a DWM at slightly different speeds in different directions and
different  frequencies,  causing  adispersion  error[42],  which
can  be  controlled  and  reduced  to  some  extent  by  means  of
careful design of the mesh topology or by using interpolated
meshes and frequency warping methods [18], [43].
Accurate  modeling  with  DWMs  requires  mesh  topologies
with a very fine resolution (e.g.≈10
7
junctions for a room of
size4×6×3m [26]). That makes the computational load and
the  amount  of  memory  required  prohibitively  high  for  real-
time  operation,  especially  for  large  rooms.  These  drawbacks
motivated  the  work  of  Karjalainenet  al.reported  in  [26],
which is reviewed in the next subsection.
Fig.  3.Conceptual  depiction  of  one  of  the  DWN  topologies  proposed
by  Karjalainen  et  al.  as  seen  by  an  observer  above  the  simulated  enclosure
(in  this  case  a  2D  rectangular  room)  [26].  The  solid  black  lines  denote  the
bidirectional delay lines interconnecting scattering nodes. The scattering nodes
are  denoted  by  theAblocks,  whereAis  the  lossless  scattering  matrix.
The  dash-dotted  lines  denote  the  unidirectional  absorptive  delay  lines.  The
dotted line denotes the line-of-sight (LOS) component. The solid arcs around
junctions  denote  loaded  self-connections  implementing  losses.  Please  note
that  while  this  figure  represents  the  case  of  a  2D  rectangular  room,  all  the
simulations in this paper use 3D rectangular rooms.
D.  Reduced Digital Waveguide Meshes
In  order  to  lower  the  computational  complexity  of  DWM
models,  Karjalainenet  al.considered  coarse  approximations
of  room  response  synthesis  via  sparse  DWM  structures  [26].
One  such  structure  is  shown  in  Fig.  3.  In  this  network,  the
sound source is connected via unidirectional absorbing delay
lines  to  scattering  junctions.  These  junctions  are  positioned
at  the  locations  where  first-order  reflections  impinge  on  the
walls.  This  ensures  that  delays  of  first-order  reflections  are
rendered accurately. The junctions are connected via bidirec-
tional delay lines with the microphone, which is also modeled
as a scattering junction contributing to the energy circulation
in  the  network.  The  line-of-sight  component  is  modeled  by
a  direct  connection  between  the  source  and  the  microphone.
Additional bidirectional delay lines parallel to the wall edges
are included to better simulate room axial modes [44]. All the
wall junctions are connected in a ring topology.
In  order  to  model  losses  at  the  walls,  it  appears  that  a
combination of junction loads and additional self-connections
is  used.  However,  implementation  details  are  not  given,  and
the  authors  state  that  the  network  required  careful  heuristic
tuning  [26].  For  these  reasons  the  results  are  difficult  to
replicate.
In  the  next  section,  we  describe  a  structure  which  renders
the direct path and first-order early reflections accurately both
in time and amplitude, while producing progressively coarser
approximations of higher order reflections and late reverbera-
tion.  The  proposed  method  has  parameters  that  are  inherited
directly  from  room  geometry  and  absorption  properties  of
wall  materials,  and  thus  does  not  require  tuning.  In  order  to
distinguish  the  proposed  structure  from  the  ones  considered
DE SENAet al.:  EFFICIENT SYNTHESIS OF ROOM ACOUSTICS VIA SCATTERING DELAY NETWORKS5
Fig.  4.Conceptual  depiction  of  the  SDN  reverberator.  The  solid  black
lines  denote  bidirectional  delay  lines  interconnecting  the  SDN  wall  nodes.
The  SDN  wall  nodes  are  denoted  by  theSblocks,  whereSis  the  lossy
scattering  matrix.  The  dash-dotted  lines  denote  unidirectional  absorptive
delay  lines  connecting  the  source  to  the  SDN  nodes.  The  dashed  lines
denote  unidirectional  absorptive  delay  lines  connecting  the  SDN  nodes  to
the microphone. The dotted line denotes the direct-path component.
by  Karjalainenet  al.in  [26],  and  because  of  the  importance
of the scattering operation, we refer to the proposed structures
asscattering delay networks(SDNs).
III.  SCATTERINGDELAYNETWORKS
The  aim  of  the  SDN  structure  is  to  simulate  the  acoustics
of an enclosure using a minimal topology which would ensure
that each significant reflection in the given space has a corre-
sponding reflection in the synthesized response. This requires
representing each significant reflective surface using one node
in a fully connected mesh topology. The concept is illustrated
in Fig. 4. For clarity, considerations in this paper will pertain to
rectangular empty rooms, however all the presented concepts
can  be  extended  in  a  straightforward  manner  to  arbitrary
polyhedral  spaces  with  additional  reflective  surfaces  in  their
interior.
As  shown  in  Fig.  4,  the  network  consists  of  a  fully  con-
nected  DWN  with  one  scattering  node  for  each  wall.  This
network is minimal in the sense that the removal of any of the
nodes or paths would make it impossible to model a significant
subset of reflections which would arise in the space.
The  source  is  connected  to  the  scattering  nodes  via  uni-
directional  absorbing  lines.  As  opposed  to  the  reverberators
proposed  in  [26],  the  microphone  node  is  a  passive  element
that  does  not  participate  in  the  energy  recirculation,  hence
scattering  nodes  are  connected  to  the  microphone  via  unidi-
rectional  absorbing  lines  and  no  energy  is  injected  from  the
microphone node back to the network.
Early reflections are known to strongly contribute to one’s
perception  of  the  size  and  spaciousness  of  a  room  [45].  For
this  reason,  nodes  are  positioned  on  the  walls  at  locations
of  first-order  reflections.  Delays  and  attenuation  of  the  lines
connecting the nodes, source, and microphone are set so that
(a)(b)
(c)(d)
Fig. 5.   Examples of approximations generated by SDN for four second-order
reflections. The solid black line represents the actual path of the second-order
reflection, while the dashed line is the corresponding path within the SDN.
first-order  reflections  are  rendered  accurately  in  their  timing
and energy.
Second-order reflections are approximated by corresponding
paths  within  the  network.  This  is  illustrated  in  Fig.  5.  It  can
be observed from the figure that the accuracy of second-order
reflections depends on the particular reflection, but nonetheless
the delays of the approximating paths are similar to the actual
ones. As the reflection order increases, coarser approximations
are made. Thus, the proposed network behaves equivalently to
geometric-acoustic  methods  up  to  the  first-order  reflections,
while  providing  a  gracefully  degrading  approximation  for
higher-order ones.
Precise details of the SDN design are given below.
Scattering   nodes:Each   node   is   positioned   on   a   wall
of  the  modeled  enclosure.  The  nodes  are  positioned  at  the
locations where the first-order reflections impinge on the walls.
These  locations  are  straightforward  to  calculate  for  simple
geometries,  e.g.  convex  polyhedra.  The  nodes  carry  out  a
scattering  operation  on  the  inputs  from  the  otherKnodes,
p
+
, to obtain the outputs asp
−
[n] =Sp
+
[n], whereSis the
K×K(not necessarily lossless) scattering matrix. Rectangular
rooms, which are used in the following, correspond toK= 5.
Other geometries whereKis a power or2are computationally
6IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 23, NO. 9, SEPTEMBER 2015
convenient since it can be shown that they lead to a multiplier-
free realization [11].
The  scattering  matrixSgoverns  how  energy  circulates
within  the  network.  Since  each  node  is  associated  with  a
wall,  it  describes  how  energy  is  exchanged  between  walls.
Furthermore, when incident wavesp
+
are scattered from the
wall, a certain amount of energy is absorbed. A macroscopic
quantity that describes material absorption sufficiently well for
most synthesis applications is the random-incidence absorption
coefficient  [13].  This  coefficient,  specified  by  the  ISO  354
standard, is known for a variety of materials [13].
The  wall  absorption  effect  can  be  expressed  in  the  most
general form asp
−∗
Yp
−
= (1−α)p
+∗
Yp
+
, whereαis the
wall absorption coefficient, which is equivalent to
S
∗
YS= (1−α)Y.(10)
By  expressingSasS=βA,  whereβ=
√
1−α,  the
relationship  in  (10)  becomes  equivalent  toA
∗
YA=Y,  i.e.
the  scattering  matrixSis  the  product  of  the  wall  reflection
coefficientβand a lossless scattering matrixA. As in DWNs,
ifAis selected as given by (5) or (6), the propagating variables
have  a  physical  interpretation  as  the  pressure  or  root-power
waves  in  a  network  of  acoustic  tubes.  Other  non-physical
choices ofAare possible, as long as the lossless condition is
satisfied; these are discussed in Section IV-C.
As  will  be  shown  in  Section  V-A,  setting  the  scattering
matrix  asS=βAresults  in  a  rate  of  energy  decay  that  is
consistent with the well-known Sabine and Eyring formulas as
well as with the results of the IM. This can be attributed to the
fact that both the average time between successive reflections
(i.e.  themean  free  path)  and  the  energy  loss  at  the  walls  in
the virtual SDN network are close to the ones observed in the
corresponding physical room.
The   absorption   characteristic   of   most   real   materials   is
frequency-dependent.  This  can  be  modeled  by  using  a  more
general  scattering  matrixS(z)  =H(z)A,  whereH(z)  =
diag{H(z),...,H(z)}, andH(z)is a wall filter. The absorp-
tion  coefficients  in  consecutive  octave  bands,  for  a  range  of
materials,  are  reported  in  [13].  Standard  filter  design  tech-
niques  can  be  used  to  fit  the  frequency  responseH(e
jω
)to
these tabulated values. Minimum-phase IIR filters are particu-
larly convenient in this context as they reduce computational
load  without  significantly  affecting  the  phase  of  simulated
reflections [46].
As  in  conventional  DWMs,  the  pressure  at  the  SDN  node
is  a  function  of  the  incoming  wave  variables,p
+
i
[n],  from
neighboring  nodes  and  the  pressure,p
S
[n],  injected  by  the
source. That is modeled according to (7), and it is illustrated
in  Fig.  6.  Other  details  of  source-to-node  connections  are
discussed next.
Source-to-node  connection:The  input  to  the  system  is
provided  by  a  source  node  connected  to  SDN  nodes  via
unidirectional absorbing delay lines (see Fig. 6).
The  delay  of  the  line  connecting  the  source  atx
S
and  the
SDN  node  positioned  atx
k
is  given  by  the  corresponding
propagation  delayD
S,k
=bF
s
‖x
S
−x
k
‖/cc,  wherecis
the  speed  of  sound  andF
s
is  the  sampling  frequency.  The
Fig. 6.Connection between the source node and an SDN node.
Fig. 7.Two interconnected SDN nodes.
attenuation  due  to  spherical  spreading  (1/rlaw)  is  modeled
as
g
S,k
=
1
‖x
S
−x
k
‖
.(11)
Source directivity is another important simulation parameter
in room acoustic synthesis. The sparse sampling of the simu-
lated enclosure prohibits the simulation of source directivity in
detail.  However,  a  coarse  approximation  can  be  incorporated
by weighting the outgoing signals byΓ
S
(θ
S,k
), whereΓ
S
(θ)
is  the  source  directivity,  andθ
S,k
is  the  angle  between  the
source  reference  axis  and  the  line  connecting  the  source  and
k-th node. An alternative approach consists of using an average
of  the  directivity  pattern  in  some  angular  sector.  It  should
be  noted  that  it  is  possible  to  simulate  frequency-dependent
characteristics of source directivity using short, variable linear
or  minimum-phase  filters.  However,  in  order  to  keep  the
exposition in this article clear it is assumed that the directivity
patterns are independent of frequency and can be modeled as
simple gains.
Node-to-node  connection:The  connections  between  the
SDN  nodes  consist  of  bidirectional  delay  lines  modeling  the
propagation path delay as shown in Fig. 7. Additional low-pass
filters can be inserted into the network at this point to model
the  frequency-dependent  characteristic  of  air  absorption,  as
proposed by Moorer in the context of FDNs [47].
The delays of the lines connecting the nodes are determined
by their spatial coordinates. Thus, the delay of the line between
a node at locationx
k
and a node atx
m
isD
k,m
=bF
s
‖x
k
−
x
m
‖/cc.
Node-to-microphone  connection:Each node is connected
to  the  microphone  node  via  a  unidirectional  absorbing  delay
line. The signal extracted from the junction,p
e
[n], is a linear
combination  of  outgoing  wave  variables,p
e
[n] =w
T
p
−
[n],
wherep
−
[n]is the wave vector after the wall filtering opera-
DE SENAet al.:  EFFICIENT SYNTHESIS OF ROOM ACOUSTICS VIA SCATTERING DELAY NETWORKS7
Fig. 8.   Connection between an SDN node and the receiver/microphone node.
tion, as shown in Fig. 8.
In  the  physical  case  whereAis  in  the  form  (5)  or  (6),
the  outgoing  signal  to  the  microphone  is  taken  as  the  node’s
pressure, as given by equations (4) and (7):
p
e
[n] =
2
〈1,y〉
y
T
p
−
[n].(12)
In  the  non-physical  case,  various  choices  are  available  for
extracting a signal from the junction. The only condition that
the weightswneed to satisfy is that the cascade of pressure
injection, scattering and extraction does not alter the amplitude
of first-order reflections. Since the incoming wave vectorp
+
for  a  first-order  reflection  with  amplitudep
S
is  given  by
p
+
= (
p
S
2
1)(see Fig. 6) and sincep
−
=Sp
+
, this condition
can be written asw
T
S
(
p
S
2
1
)
=βp
S
or, equivalently, as
w
T
A1= 2.(13)
A  possible  choice  forwis  a  constant  vector,  which  is
computationally  convenient  since  it  requires  a  single  multi-
plication.  In  this  case,  the  constraint  (13)  yields  the  unique
solutionw=
2
1
T
A1
1.
The  delay  from  thek-th  SDN  node  to  the  microphone
node  isD
k,M
=bF
s
‖x
k
−x
M
‖/cc.  As  with  the  source
directivity,  the  microphone  directivity  is  also  modeled  using
a  simple  gain  elementΓ
M
(θ
k,M
),  whereΓ
M
(θ)is  the  mic-
rophone directivity pattern andθ
k,M
is the angle between the
microphone acoustical axis and thek-th node. This approach
ensures  that  the  microphone  is  emulated  correctly  for  the
directions associated to the first-order reflections. As with the
source-node connections, the microphone directivity can also
be  modeled  using  short,  variable  linear  or  minimum-phase
filters.  However,  simple  gain  elements  are  preferred  in  this
article for clarity of presentation. Similarly, as with the source-
node connection, an alternative approach consists of using an
average of the directivity pattern in some angular sector.
The attenuation coefficient is set as
g
k,M
=
1
1 +
‖x
k
−x
M
‖
‖x
S
−x
k
‖
,(14)
such that, using (11),
g
S,k
×g
k,M
=
1
‖x
S
−x
k
‖+‖x
k
−x
M
‖
,(15)
which yields the correct attenuation for the first-order reflec-
tion according to the1/rlaw. Notice that the above choice of
g
k,M
andg
S,k
is not unique in satisfying the constraint (15).
The attenuation can in fact be distributed differently between
the source-to-node and node-to-microphone branches but with
little impact on overall energy decay.
IV.  PROPERTIES OFSDNS
A.  Transfer function
The  block  diagram  of  an  SDN  system  is  shown  in  Fig.  9.
In the figure,
Γ
S
= [Γ
S
(θ
S,1
),Γ
S
(θ
S,2
),...,Γ
S
(θ
S,K+1
)]
T
,(16)
D
S
(z) =diag
{
z
−D
S,1
,z
−D
S,2
,...,z
−D
S,K+1
}
,(17)
G
S
=diag{g
S,1
,g
S,2
,...,g
S,K+1
},(18)
are the source directivity vector, the source delay matrix, and
the source attenuation matrix, respectively. The corresponding
quantities  associated  with  the  microphoneΓ
M
,D
M
(z)and
G
M
are  defined  as  in  (16),  (17)  and  (18),  by  substitutingS
withM. Further,
U=diag
{
1,...,1
︸
︷︷︸
K+1
}
,(19)
D
f
(z) =diag
{
z
−D
1,2
,...,z
−D
K,K+1
}
,(20)
H(z) =diag
{
H
1
(z),...,H
1
(z)
︸︷︷︸
K
,...,H
K+1
(z)
}
,(21)
are the block-diagonal matrix that distributes source signals to
input wave variables, the inter-node delay matrix, and the wall
absorption matrix, respectively. The block diagonal scattering
matrixAand the block diagonal weight matrixWare defined
as
A=diag{A
1
,A
2
,... ,A
K+1
},(22)
W=diag
{
w
T
1
,...,w
T
K+1
}
,(23)
whereA
k
andw
k
are  thei-th  node’s  scattering  matrix  and
pressure extraction weights, respectively. While these variables
can in general be different for each node, in all the simulations
presented  in  this  paper  they  are  selected  to  be  equal,A
1
=
···=A
K+1
andw
1
=···=w
K+1
.
Finally,  in  Fig.  9,g
S,M
andz
−D
S,M
are  the  line-of-sight
attenuation  and  delay,  respectively,  andPis  a  permutation
matrix whose elements are  determined based on the network
topology.   Due   to   the   underlying   node   connectivity   being
bidirectional,  this  permutation  matrix  is  symmetric.  For  the
simplest case of a three-dimensional enclosure with a rectan-
gular shape, the permutation matrix takes the formP=δ
i,f(j)
,
whereδ
i,j
is the Kronecker delta,
f(i) = ((6i−((i−1))
N
−1))
N(N−1)
+ 1,(24)
and((·))
N
is  the  modulo-Noperation.  Inspection  of  Fig.  9
reveals that the system output can be expressed as
Y(z) =Γ
T
M
D
M
(z)G
M
Wq(z) +
gz
−D
S,M
X(z)(25)
whereg=g
S,M
Γ
S
(θ
S,M
) Γ
M
(θ
M,S
),  andq(z)is  the  state
vector, given by
q(z) =
1
2
[
I−
H(z)APD
f
(z)
]
−1
H(z)AUG
S
D
S
(z)Γ
S
X(z).
(26)
8IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 23, NO. 9, SEPTEMBER 2015
Fig. 9.Block diagram of the SDN reverberator.
The transfer function can therefore be expressed as
H(z) =
1
K
k
T
M
(z)
[
A
T
H
−1
(z)−PD
f
(z)
]
−1
k
S
(z)+
gz
−D
S,M
,
(27)
wherek
T
M
(z)=Γ
T
M
D
M
(z)G
M
Wandk
S
(z)=
UG
S
D
S
(z)Γ
S
.
It may be observed that, unlike FDN reverberators, relevant
acoustical aspects, such as the direct path and reflection delays,
are  modeled  explicitly,  allowing  complete  control  of  source
and microphone positions and their directivity patterns.
Expressing  the  system  transfer  function  as  given  above
allows for a complete demarcation of the directional properties
and  positions  of  the  source  and  microphone,  wall  absorption
characteristics  and  room  shape  and  size.  This  is  especially
useful in keeping computational cost low for cases where only
a single aspect, such as source orientation, changes.
B.  Stability
The  stability  of  the  SDN  follows  from  the  fact  that  its
recursive  part,  i.e.  the  backbone  formed  by  the  SDN  nodes,
is  a  fully  connected  DWN.  The  stability  of  lossless  DWNs
is,  in  turn,  guaranteed  by  the  fact  that  the  network  has  a
physical  interpretation  as  a  network  of  acoustic  tubes  [48],
[10]. Indeed, the ideal physical system’s total energy provides
a  Lyapunov  function  bounding  the  sum-of-squares  in  the
numerical simulation (provided one uses rounding toward zero
or  error  feedback  in  the  computations)  [48].  Furthermore,
the  network  conserves  its  stability  when  losses  due  to  wall
absorption  are  included  at  the  SDN  nodes  since  the  physical
pressure  (i.e.  the  sum  of  incoming  and  outgoing  pressure
waves) is then always reduced relative to the lossless case.
C.  Lossless Scattering Matrices
In  this  section,  we  explore  possible  choices  for  lossless
scattering  matrices  and  discuss  their  implications.  First  we
present  a  complete  parametrization  of  real  lossless  matrices.
The parameterization is an immediate corollary of the follow-
ing theorem.
Theorem 1.A real square matrixAis diagonalizable if and
only if it has the form
A=T
−1
ΛT,(28)
whereTis a real invertible matrix, andΛis a block diagonal
matrix  consisting  of1×1blocks  which  are  real  eigenvalues
ofA, and2×2blocks of the form
[
0−r
i
r
i
2r
i
cos(θ
i
)
]
,
wherer
i
e
jθ
i
are  complex  eigenvalues  ofAwhich  appear  in
pairs with their conjugates.
The theorem is proved in the appendix.
Corollary 2.A real square matrix is lossless if and only if it
has the form given by Theorem 1 with all eigenvalues on the
unit circle.
The corollary follows from the fact that a square matrix is
lossless if and only if it is diagonalizable and all its eigenvalues
are on the unit circle [10].
Within  the  space  of  lossless  matrices,  a  large  degree  of
leeway  is  left  for  pursuing  various  physical  or  perceptual
criteria.  This  can  be  achieved,  for  instance,  by  finding  a
lossless  matrixAwhich  minimizes  the  distance  from  a  real
matrixDthat reflects some sought-after physical or perceptual
properties. The design then amounts to constrained optimiza-
tion:
min
A
‖A−D‖
2
F
,(29)
where‖·‖
F
denotes the Frobenius norm, under the constraint
thatAhas  the  form given  in  Corollary  2. This  most  general
case  may,  however,  involve  optimization  over  an  excessive
number of parameters. If we restrict the optimization domain
to  orthogonal  matrices,  solutions  can  be  found  without  the
need  for  numerical  procedures.  In  particular,  the  following
Theorem result holds:
Theorem 3.The solution to the following optimisation prob-
lem
argmin
A
‖A−D‖
2
F
,A
T
A=I(30)
is given byA=UV
T
, whereUandVare respectively the
matrices of left and right singular vectors ofD.
A proof of this result can be found in [49].
Orthogonal   matrices   which   are   also   circulant   have   the
interpretation  of  performing  all-pass  circulant  convolution  of
the  incoming  variables,  which  as  discussed  below,  reduces
computational  complexity  of  the  scattering  operator.  Further-
more,  if  a  certain  distribution  of  (unit-norm)  eigenvalues  is
sought, the associated circulant matrix can be found by means
of  a  single  inverse  fast  Fourier  transform  (FFT)  [10].  An  in-
depth study of such scattering matrices in the context of DWN
reverberators has been presented in [10].
DE SENAet al.:  EFFICIENT SYNTHESIS OF ROOM ACOUSTICS VIA SCATTERING DELAY NETWORKS9
Householder  reflection  matrices,  given  byA=  2vv
T
−
I,‖v‖= 1, are the subclass of orthogonal scattering matrices
commonly  used  in  the  context  of  DWN  reverberators.  As
discussed in Section II-B, they enable a physical interpretation
of the propagating variables as normalized pressure waves in
a  network  of  acoustic  tubes.  The  optimization  problem  that
minimizes  the  distance  from  a  targeted  scattering  matrix  in
this case can be expressed as
argmin
v
‖(2vv
T
−I)−D‖
2
F
,‖v‖= 1,(31)
the solution of which is given by the following theorem.
Theorem 4.The solution to the optimization problem in (31)
is  the  singular  vector  corresponding  to  the  largest  singular
value of matrixD+D
T
.
The theorem is proved in the appendix.
The  isotropic  scattering  matrix,i.e.the  particular  case  of
a  Householder  reflection  obtained  forv=1/
√
K,  can  be
physically  interpreted  as  the  scattering  matrix  which  takes  a
reflection  from  one  node  and  distributes  its  energy  equally
among  all  other  nodes.  This  is  the  only  orthogonal  matrix
which has this property, as stated by the following theorem.
Theorem  5.IfAis  an  orthogonal  matrix  which  scatters
the energy from each incoming direction uniformly among all
other directions, then it must have the formA=±
2
K
11
T
−I.
The  theorem is  proved  in the  appendix.  The  isotropic  matrix
thus satisfies some optimality criteria and, as discussed below,
allows for fast implementation. We will see in Section V-B that
its  special  structure  leads  to  a  fast  build  up  of  echo  density
which is a perceptually desirable quality [29].
These  different  choices  of  scattering  matrices  have  their
implications on computational complexity. The computational
complexity  of  the  matrix-vector  multiplication  using  gen-
eral  lossless  and  orthogonal  matrices  isO[K
2
]operations.
Circulant  lossless  matrices  require  onlyO[Klog(K)]oper-
ations  [10].  The  computational  complexity  associated  with
matrices  which  have  the  form  of  Householder  reflections  is
further reduced toO[K]. Among general Householder reflec-
tions, which require2K−1additions and2Kmultiplications,
the isotropic scattering matrix only requires2K−1additions
and1multiplication.  Among  lossless  matrices,  the  ones  that
require the least operations are permutation matrices. However,
we will see in Section V-B that permutation matrices lead to
an insufficient echo density.
D.  Interactivity and multichannel auralization
Interactive  operation  of  the  SDN  reverberator  is  accom-
plished by updating the model to reflect changes in the posi-
tions and rotations of the source and microphone. This requires
readjusting  the  positions  of  the  wall  nodes,  and  updating  the
delay  line  lengths  and  gains  accordingly.  It  was  observed  in
informal listening tests that updating the delay line lengths did
not cause audible artifacts as long as microphone and source
speeds are within reasonable limits.
051015
10
5
10
6
10
7
10
8
10
9
Structure order
FLOPS
SDN
FDN
Fig.  10.Comparison  of  computational  complexity  for  SDN  and  FDN  as
a  function  of  the  structure  order,  i.e.  the  size  of  the  feedback  matrix  for
FDN and the number of neighboring nodesKfor SDN (notice that the case
of  rectangular  rooms  corresponds  toK= 5).  The  sampling  frequency  is
F
s
= 44100Hz.
The   proposed   model   allows   approximate   emulation   of
coincident   and   non-coincident   recording   formats.   Coinci-
dent  formats  (e.g.  Ambisonics  [21],  higher-order  ambisonics
(HOA)  [50],  vector-base  amplitude  panning  (VBAP)  [51])
can  be  easily  employed  by  adjusting  the  microphone  gains
Γ
M,k
(θ)appropriately.
Non-coincident  formats  (e.g.  [52],  [53])  can  be  emulated
by  considering  a  separate  SDN  for  each  microphone.  This
results in a higher inter-channel decorrelation, which is what
would actually occur in real recordings. However, the overall
computational   load   also   increases.   If   simulation   speed   is
critical, an alternative approach would be to share the same set
of  wall  nodes  among  all  the  SDNs,  while  creating  dedicated
node-to-microphone connection lines for each microphone.
E.  Computational load
This  section  presents  an   analysis  of  the  computational
complexity  of  the  proposed  model  in  comparison  to  the  two
conceptually  closest  technologies,  FDN  and  the  IM.  We  use
the  number  of  floating  point  operations  per  second  as  an
approximate indicator of the overall computational complexity.
Furthermore, to simplify calculations we make the assumption
that  additions  and  multiplications  carry  the  same  cost.  This
approximation is motivated by the progressive convergence of
computation time of various operations in modern mathemat-
ical processing units.
The   number   of   floating   point   operations   per   second
(FLOPS)   performed   by   an   SDN   can   be   calculated   as
F
s
[
2K
3
+ (P+ 2)K
2
+K+ 1
]
,  wherePis  the  number
of  operations  required  by  each  wall  filter  for  each  sample.
Consider now an FDN with aQ×Qfeedback matrix. From
inspection  of  Fig.  1  the  computational  complexity  can  be
shown to beF
s
[
2Q
2
+ (P+ 3)Q+ 1
]
FLOPS. Fig. 10 shows
a  comparison  of  the  computational  complexity  of  SDNs  and
FDNs.  In  this  figure,  the  x-axis  denotes  the  structure  order,
which  we  define  as  the  number  of  neighboring  nodesKfor
SDN  and  the  size  of  the  feedback  matrixQfor  FDN.  The
10IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 23, NO. 9, SEPTEMBER 2015
sampling frequency is set toF
s
= 44100Hz, and the filtering
step consists of a simple frequency independent gain (P= 1)
for both SDN and FDN. It may be observed that for the typical
case of a 3D rectangular room (K= 5), SDN has around the
same  computational  complexity  of  an  FDN  with  a12×12
feedback  matrix.  More  specifically,  SDN  forK=  5has  a
complexity  of14.6MFLOPS,  while  FDN  forQ= 12has  a
complexity  of14.9MFLOPS.  Notice  that  the  computational
complexity of both SDN and FDN can be reduced significantly
by  using  efficient  lossless  matrices  of  the  type  discussed  in
Section IV-C, e.g. Householder and circulant lossless matrices.
Consider  now  the  computational  cost  of  the  IM  method
for  rectangular  rooms  in  its  original  time-domain  implemen-
tation [15]. In the IM, each image source requires10floating-
point operations
1
to calculate the time index and1/rattenua-
tion of the image, and15floating-point operations to calculate
its  attenuation  due  to  wall  absorption.  This  amounts  to25
floating-point  operations  in  total  for  each  image  source.  The
number of image sources contained in an impulse response of
lengthT
60
seconds  is  approximately  equal  to  the  number  of
room cuboids that fit in a sphere with radiuscT
60
meters. This
gives a computational complexity of around
25
⌈
4
3
π(T
60
c)
3
L
x
L
y
L
z
⌉
FLOPS,(32)
whereL
x
,L
y
andL
z
are the room dimensions. Here, we are
implicitly  assuming  that  the  entire  RIR  is  calculated  using
the IM. While this ensures a fair comparison in terms of the
other properties assessed in the next section, it should be noted
that, in order reduce the complexity, most auralization methods
use  the  IM  only  to  simulate  the  early  reflections.  This  is
especially so in case of non-rectangular rooms, which require
a  considerably  larger  amount  of  memory  and  computational
power than equation (32) [16]. The rest of the RIR is usually
generated  using  lightweight  statistical  methods,  e.g.  random
noise with appropriate energy decay [54].
Once  the  RIR  has  been  generated,  it  has  to  be  convolved
with  the  input  signal  to  obtain  the  reverberant  signal.  In  the
case  of  real-time  applications,  this  can  be  done  efficiently
using  the  overlap-add  method  [55].  The  method  calculates
2FFTs  of  lengthN,Ncomplex  multiplications,1inverse
FFT,  anddT
60
F
s
ereal  additions  for  each  time  frame.  In
order  for  the  circular  convolution  to  be  equal  to  the  linear
convolution,Nmust  satisfyN≥ dF
s
/F
r
e+dT
60
F
s
e−1,
whereF
r
is  the  frame  refresh  rate.  In  the  best-case  scenario
whereNis  a  power  of2,  the  asymptotic  cost  of  each  FFT
is6Nlog
2
N[55].  Furthermore,  each  complex  multiplica-
tion  requires4real  multiplications  and2additions.  Overall,
the  computational  complexity  of  the  overlap-add  method  is
F
r
(18Nlog
2
N+6N+dT
60
F
s
e−1)FLOPS. In the static case,
the FFT of the impulse response can be precomputed, and the
cost reduces toF
r
(12Nlog
2
N+ 6N+dT
60
F
s
e−1)FLOPS.
Fig.  11  compares  the  cost  of  SDN  with  the  static  and
dynamic  IM.  In  the  static  case,  both  microphone  and  source
are  not  moving.  In  the  dynamic  case,  on  the  other  hand,
1
Here  we  assume  that  exponentiations  and  square  roots  count  as  a  single
floating-point operation.
00.20.40.60.81
10
6
10
7
10
8
10
9
10
10
T60 [s]
FLOP
S
SDN
Overlap−add (static)
Overlap−add and IM (dynamic)
Fig.  11.Computational  complexity  of  SDN  in  comparison  to  overlap-add
convolution in both static and dynamic modes as a function of reverberation
time.  The  static  case  represents  the  cost  of  the  overlap-add  convolution
with  a  fixed,  precomputed  RIR.  The  dynamic  case  also  includes  the  cost
of  calculating  a  new  RIR  and  its  FFT  for  each  time  frame.  The  sampling
frequency isF
s
= 44100Hz.
microphone and/or source are moving, and the IM is run for
each frame. The refresh rate is chosen such that the buffering
delay  is  shorter  than  the  maximum  latency  of  a  half-frame
delay  between  the  video  and  audio,  as  recommended  by  the
ITU Recommendation BR.265-9 [56]. For a video running at
25frames  per  second,  this  criterion  gives  a  refresh  rate  of
F
r
= 50Hz. The room size is the same as used in Allen and
Berkley’s paper of10×15×12.5feet [15]. It may be observed
that for typical medium-sized rooms, SDN is from about10to
100times  faster  than  dynamic  IM.  SDN  is  also  significantly
faster than (overlap-add) convolution alone.
While  we  compare  the  computational  complexity  of  the
proposed algorithm with the standard overlap-add convolution,
we also acknowledge that more efficient convolution methods
have  recently  been  proposed,  e.g.  [57]–[59].  However,  the
comparison of the proposed algorithm with these new methods
is outside the scope of this article.
F.  Memory requirement
The required memory is determined by the number of taps
of  the  delay  lines.  An  upper  bound  for  memory  requirement
Qcan  be  easily  found  by  observing  that  the  length  of  each
delay line is smaller than or equal to the distance between the
two farthest points of the simulated space, giving:
Q≤(N(N−1) + 2N+ 1)
qF
s
c
Rbits,(33)
whereqis  the  number  of  bits  per  sample,  andRis  the
maximum  distance  between  any  two  points  in  the  simulated
space.  The  value  ofRin  the  case  of  a  rectangular  room  is
R=
√
L
2
x
+L
2
y
+L
2
z
.  For  the  more  general  case,Ris  the
diameter of the bounding sphere of the room shape.
Observe  in  (33)  thatQscales  linearly  with  the  room  size.
For  a  cubic  room  with  a5m  edge,F
s
= 40kHz,  andq=
32bytes  per  sample,  the  memory  requirement  is  less  than
170kB, which is negligible for virtually every state-of-the-art
platform.
DE SENAet al.:  EFFICIENT SYNTHESIS OF ROOM ACOUSTICS VIA SCATTERING DELAY NETWORKS11
0246810
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
Room edge [m]
T60 [s]
SDN
IM
Sabine equation
Eyring equation
Fig.  12.Values  of  reverberation  timeT
60
as  a  function  of  the  edge  of  a
cubic room. The wall absorption isα= 0.5, and both microphone and source
are positioned in the volumetric center of the room.
V.  ASSESSMENT
This section presents the results of assessments of SDNs in
terms of perceptually-based objective criteria.
As  discussed  in  previous  sections,  first-order  reflections
are  rendered  correcty  both  in  timing  and  amplitude  by  con-
struction.  Another  cue  important  for  the  perception  of  room
volume  and  materials  is  the  reverberation  time  [45].  Sec-
tion  V-A  presents  an  evaluation  of  the  reverberation  time
both in frequency-independent and frequency-dependent cases.
Section  V-B  focuses  on  the  time  evolution  of  echo  density,
which  is  related  to  the  perceived  time-domain  texture  of
reverberation [29].
Here,  the  IM  is  used  as  a  reference  since  it  is  the  closest
method  among  physical  room  models.  More  specifically,  we
use a C++ version of Allen and Berkley’s original time-domain
implementation [15].
A.  Reverberation time
The parameter most commonly used to quantify the length
of reverberation isT
60
, which is defined as the time it takes for
the room response to decay to 60 dB below its starting level.
In this section, theT
60
of the SDN network is compared with
two well-known empirical formulas proposed by Sabine [60]
and Eyring [61]:
T
60,Sab
=
0.161V
∑
i
A
i
α
i
,(34)
T
60,Eyr
=−
0.161V
(
∑
i
A
i
) log
10
(1−
∑
i
A
i
α
i
/
∑
i
A
i
)
,(35)
whereVis  the  room  volume,A
i
andα
i
are  the  area  and
absorption coefficient of thei-th wall, respectively.
1)  Frequency-independent  wall  absorption:Cubic  rooms
with  different  volumes  and  uniform  frequency-independent
absorption are simulated. In order to maintain the experimental
conditions  across  different  room  sizes,  both  the  source  and
the  microphone  are  placed  at  the  volumetric  center  of  the
00.20.40.60.81
10
−2
10
−1
10
0
10
1
Absorption coefficient α
T60 [s]
Velvet curtainAudience (0.75 pers./msq)Fissured ceiling tiles
Cotton carpetBricks with rough finish
SDN
IM
Sabine equation
Eyring equation
Fig. 13.   Reverberation time values as a function of the absorption coefficient
αfor SDN, IM, and Sabine and Eyring predictions. The simulated enclosure
is  a  cube  with5m  edge.T
60
values  are  averaged  across  10  source  and
microphone  random  positions.  The1kHz  absorption  coefficient  of  various
materials as measured by Vorl
̈
ander in [13] are reported at the bottom of the
plot as reference.
room. Furthermore, the line-of-sight component was removed,
as  suggested  in  the  ISO  3382  standard  [62]  for  measuring
the  reverberation  time  in  small  enclosures.  In  Fig.  12,  the
reverberation  time  is  shown  as  a  function  of  the  edge  length
for a room absorption coefficientα= 0.5. It may be observed
that  the  SDN  generates  room  impulse-responses  which  have
reverberation times that increase linearly with the edge length.
This is due to the larger average distance between the nodes,
which in turn increases the mean free path of the structure.
TheT
60
values corresponding to the SDN reverberator are
between the predictions given by Sabine and Eyring’s formulas
and are nearly identical to the ones produced by the IM. The
latter result may seem surprising if one considers that the SDN,
as  opposed  to  the  IM,  does  not  include  attenuation  due  to
spherical spreading (except for the initial first-order reflections
and microphone taps). This apparent inconsistency is resolved
intuitively  by  observing  that  spherical  spreading  is  a  lossless
phenomenon: In the IM, the quadratic energy decrease (1/r
2
)
is  compensated  by  the  quadratic  increase  of  the  number  of
contributing image sources over time, and similarly to that, in
DWN and SDN, “plane waves” are scattered losslessly at each
node, thus conserving the energy.
In  Fig.  13,  the  reverberation  time  is  shown  as  a  function
of the absorption coefficientα. The enclosure was taken as a
cubic room with a5m edge, and results were averaged across
10 pairs of source-microphone positions. The coordinates were
taken  from  a  uniform  random  distribution  and  satisfied  both
requirements  set  in  [62]:  The  microphone  was  at  least1m
away  from  the  nearest  wall,  and  the  distance  between  the
source and microphone was at least
d
min
= 2
√
V
cT
est
,(36)
whereT
est
is a coarse estimation of the reverberation time. In
these simulations,T
est
was set using Sabine’s formula.
12IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 23, NO. 9, SEPTEMBER 2015
10
2
10
3
10
4
0.1
0.2
0.3
0.4
0.5
0.6
Frequency [Hz]
T60 [s]
SDN
Sabine equation
Fig. 14.   Comparison of reverberation time in different octave bands for SDN
and Sabine’s formula prediction.
It  may  be  observed  that  for  absorption  coefficients  higher
than  aroundα=  0.4,  SDN  and  the  IM  are  nearly  identical
and  are  both  between  Sabine  and  Eyring’s  formulas.  For
high  absorption  coefficients,  both  SDN  and  the  IM  approach
Eyring’s  formula,  which  is  known  to  give  more  accurate
predictions in that region [61]. For low absorption coefficients,
SDN  and  the  IM  produce  reverberation  times  longer  than
Sabine and Eyring’s formulas, with SDN being closer to both.
2)  Frequency-dependent wall absorption:Fig. 14 shows the
result  of  a  simulation  where  all  walls  mimic  the  frequency-
dependent  absorption  of  cotton  carpet.  The  filtersH
i
(z)are
all  set  to  be  equal  to  a  filterH(z)which  was  implemented
as  a  minimum-phase  IIR  filter  with  coefficients  optimized
by  a  damped  Gauss-Newton  method  to  fit  the  absorption
coefficients reported by Vorl
̈
ander in [13]. This procedure gave
H(z) =
0.6876−1.9207z
−1
+ 1.7899z
−2
−0.5567z
−3
1−2.7618z
−1
+ 2.5368z
−2
−0.7749z
−3
.
The  source  and  microphone  were  positioned  on  the  diagonal
of a cubic room with5m edge. More specifically, they were
positioned  on  the  diagonal  at  a  distance  ofd
min
=  2.96m
from the center, as specified using (36).
In  Fig.  14  the  wall  filter  response  is  plotted  together  with
the  corresponding  Sabine  predictions  in  (34),  which  for  the
given room becomes
T
60,Sab
=
0.161V
∑
i
A
i
α
i
(ω)
=
0.161·5
6α(ω)
=
0.161·5
6(1−|H(e
jω
)|
2
)
.
(37)
The simulated RIR was fed into an octave-band filter bank, and
T
60
values were calculated for each octave-band. As shown in
Fig. 14, these measuredT
60
are very close to Sabine’s formula
prediction,  thus  confirming  that  the  proposed  model  allows
controlling the absorption behavior of wall materials explicitly.
Note  that  if  explicit  control  of  the  reverberation  time  is
sought,  the  prediction  functions  (34)  or  (35)  can  be  inverted
to obtain neededα(ω). To this end, Eyring’s formula (35) is
preferable,  since  (34)  may  yield  non-physical  values  for  the
absorption coefficient (i.e.α >1) of some acoustically “dead”
rooms [61].
00.020.040.060.080.10.120.140.160.180.2
0
0.3
0.75
1
Time [s]
Normalized echo density
/
/
SDN Isotropic
SDN Random
SDN Permutation
IM
Fig. 15.    Time evolution of the normalized echo density for the IM and SDN
with various scattering matricesS.
B.  Echo density
The  time  evolution  of  echo  density  is  commonly  thought
to  influence  the  perceived  time-domaintextureof  reverber-
ation  [29].  In  an  effort  to  quantify  this  perceptual  attribute,
Abel and Huang defined the normalized echo density (NED),
which was found to have a very strong correlation with results
of listening tests [29]. The NED is defined as the percentage
of  samples  of  the  room  impulse  response  lying  more  than  a
standard deviation away from the mean in a given time window
compared to that expected for Gaussian noise. A NED equal
to1means  that,  within  the  considered  window,  the  number
of samples lying more than one standard deviation away from
the mean is equal to the one observed with Gaussian noise.
Fig.  15  shows  the  time  evolution  of  the  NED  obtained
with  the  IM  and  with  the  proposed  model  using  three  dif-
ferent  scattering  matrices.  The  scattering  matrices  are  (a)
the  isotropic  matrix,  (b)  a  random  orthogonal  matrix,  and
(c)  a  random  permutation  matrix.  The  random  orthogonal
matrix was obtained by setting the angles of a Givens-rotation
parametrization  of  orthogonal  matrices  [63]  at  random.  The
simulated enclosure was a rectangular room with dimensions
3.2×4.0×2.7m, and results were averaged across50random
pairs of source and microphone positions. The wall gains were
set  toβ=−
√
0.9(wall  absorption  ofα=  0.1),  with  the
negative  sign  being  chosen  in  order  to  obtain  a  zero-mean
reverberation tail with the IM.
Fig.  15  shows  that  the  build-up  of  echo  density  of  SDN
is  very  close  to  that  of  the  IM  when  the  isotropic  scattering
matrix  is  employed.  In  particular,  the  NED  values  of0.3
and0.75,  which  were  previously  identified  as  breakpoints
dividing three perceptually distinct groups [29], are reached at
around the same delays by the two methods. Notice how the
permutation matrix fails to reach a Gaussian-like reverberation.
The random orthogonal matrix, on the other hand, does reach
a  Gaussian-like  reverberation,  but  it  takes  longer  to  achieve
the  desired  reverberation  quality  characterized  by  the0.75
breakpoint.
C.  Mode density
The  mode  density,  i.e.  the  average  number  of  resonant
frequencies  per  Hz,  is  another  important  perceptual  property
DE SENAet al.:  EFFICIENT SYNTHESIS OF ROOM ACOUSTICS VIA SCATTERING DELAY NETWORKS13
in  artificial  reverberation  [6].  In  order  to  achieve  a  natural-
sounding  reverberation,  the  mode  density  should  be  suffi-
ciently high, such that no single resonance stands out causing
metallic-sounding   artifacts.   A   threshold   for   the   minimum
mode density that is commonly used in this context is [6]:
d
min
=
T
60
6.7
,(38)
which is due to an early work of Schroeder [64].
The mode density in SDN can be calculated using consider-
ations similar to those used for FDNs [6], [34]. Assuming that
the wall filters are simple gains, i.e.H(z) =βI, and applying
the inversion lemma to the transfer function (27), the poles of
the system can be calculated as the solutions of
det
(
D
f
(z
−1
)−β
AP
)
= 0.(39)
Using  Leibniz’s  formula  for  determinants  it  is  easy  to  see
that  the  order  of  the  polynomial  in  (39)  is  equal  to  the
summation of all the delay-line lengths in the SDN backbone,
i.e.
∑
i
∑
j6=i
D
i,j
. Using the fundamental theorem of algebra,
and assuming that the poles are uniformly distributed [34], the
mode density of the SDN network can be calculated as
d
f
=
1
F
s
∑
i
∑
j6=i
D
i,j
.(40)
The  SDN  structure  satisfiesd
f
> d
min
under  most  condi-
tions of practical interest. This can be easily shown analytically
in  the  case  where  source  and  microphone  are  close  to  the
volumetric center of a cubic room with edgeL. In this case,
the length of the delay lines is approximatelyL
F
s
c
for the six
lines connecting opposite walls and
√
2
2
L
F
s
c
for the remaining
twenty four lines. The mode density is thus
d
f
=
(
6 + 24
√
2
2
)
L
c
≈23
L
c
.(41)
The conditiond
f
> d
min
is then satisfied whenever
L >
c
6.7×23
T
60
≈2.22T
60
.(42)
Since practicalT
60
values for reverberation are on the order of
a second, it follows that SDNs has a sufficient mode density
as long asL >2.22m (i.e. volume larger than around11m
3
),
which covers most cases of practical interest.
By replacingT
60
in (42) with Sabine’s approximation (34),
it  can  also  be  seen  that  the  conditiond
f
> d
min
is  satisfied
whenever  the  absorption  coefficient  is  larger  thanα >0.06,
or, equivalently,β <0.97.
In  order  to  assess  the  mode  density  in  cases  more  general
than  the  cubic  one,  Monte  Carlo  simulations  were  run  using
rectangular  rooms  with  randomly  selected  parameters.  More
specifically,  the  three  room  dimensions  were  each  drawn
from  a  uniform  distribution  between2and10meters.  The
absorption coefficient was common to all walls and was drawn
from a uniform distribution between0and1. The microphone
and  the  source  were  placed  in  positions  chosen  at  random
within  the  room  boundaries.  Out  of  1000  simulations,  SDN
provided  a  sufficient  mode  density  in  the94.7%  of  cases.
Among the cases that did not provide sufficient echo density,
the  largest  absorption  coefficient  wasα=  0.066,  which  is
largely in agreement with the result obtained above for cubic
rooms.
VI.  CONCLUSIONS ANDFUTUREWORK
This  paper  presented  a  scalable  and  interactive  artificial
reverberator termed scattering delay network (SDN). The room
is modeled by scattering nodes interconnected by bidirectional
delay lines. These scattering nodes are positioned at the points
where  first-order  reflections  originate.  In  this  way,  the  first-
order reflections are simulated correctly, while a rich but less
accurate  reverberation  tail  is  obtained.  It  was  shown  that,
according to various objective measures of perceptual features,
SDN  achieves  a  reverberation  quality  similar  to  that  of  the
IM  while  having  a  computational  load  one  to  two  orders  of
magnitude lower.
The  interested  reader  can  listen  to  SDN-generated  audio
samples that are made available as supplementary download-
able material at http://ieeexplore.ieee.org and at [65].
Several directions for future research can be envisioned on
the  basis  of  the  work  presented  in  this  paper.  The  design
of  the  backbone  network  in  Fig.  4  has  a  simple  geometrical
interpretation.  However,  the  lengths  of  the  delay  lines  in  the
backbone network can be designed using different approaches.
It is believed, in fact, that, as long as the network has a mean-
free-path  similar  to  the  one  of  the  corresponding  physical
space, the SDN will conserve its appealing properties observed
in  Sec.  V.  The  delay-line  lengths  could  be  designed,  for
instance, to minimize the average timing error of higher-order
reflections or to achieve a better fit with the modal frequencies
of the physical space. The number of SDN nodes could also
be increased. Indeed, while using a single SDN node for each
wall is the minimum to ensure that all higher-order reflections
are  modeled,  a  larger  number  of  nodes  could  be  used,  for
instance,  to  further  increase  the  modal  density  or  to  emulate
higher-order reflections  exactly. This  would, of  course, come
at the expense of an increased computational complexity.
APPENDIX
Towards  proving  Theorem  1,  we  first  establish  the  following
lemma.
Lemma 6.Two diagonalizable matrices have the same eigen-
values if and only if they are similar.
Proof  of  Lemma  6:The  sufficiency  is  a  well  know
result on similar matrices [66]. To prove the necessity, let us
consider  two  diagonalizable  matricesAandBwhich  have
the  same  eigenvalues.  SinceAis  diagonalizable,  it  follows
thatV
−1
A
AV
A
=Λ,  whereΛis  a  diagonal  matrix  andV
A
is  an  invertible  matrix.  Hence,  the  following  equality  holds:
AV
A
=V
A
Λ,  which  implies  thatΛhas  eigenvalues  ofA
on  its  diagonal.  SinceBis  also  diagonalizable  and  has  the
same  eigenvalues  asA,  it  satisfiesV
−1
B
BV
B
=Λ.  Thus,
V
−1
A
AV
A
=Λ=V
−1
B
BV
B
,  which  further  implies  that
A=
(
V
B
V
−1
A
)
−1
B
(
V
B
V
−1
A
)
,i.e.the  two  matrices  are
similar.
Using Lemma 6 we can now prove Theorem 1:
14IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 23, NO. 9, SEPTEMBER 2015
Proof  of  Theorem  1:First,  observe  thatΛis  block
diagonal,  and  that  eigenvalues  of  each  block  are  mutually
distinct. Hence, each block ofΛis diagonalizable, and there-
foreΛis itself diagonalizable [66] (over the field of complex
numbers,C). Thus,Λhas the same eigenvalues asAand is
diagonalizable.  SinceΛandAare  both  diagonalizable,  and
have the same eigenvalues, according to Lemma 6 they must
be  similar  overC.  Moreover,  since  bothΛandAare  real
and  similar  overC,  they  must  be  also  similar  overR[66],
that  is,  there  must  exist  a  real  invertible  matrixTsuch  that
A=T
−1
ΛT. On the other hand ifAhas the form given in
(28), it is diagonalizable sinceΛis diagonalizable.
Proof  of  Theorem  4:Substituting  the  definition  of  a
Householder transformation into the cost function in (31), here
termedΦ(v),  leads  with  simple  algebraic  manipulations  to
Φ(v) =
(
v
T
v−1
)
v
T
v−v
T
Dv+const. MinimizingΦ(v)
subject  tov
T
v=  1is  therefore  equivalent  to  maximising
Φ
1
(v)  =v
T
Dvunder  the  same  constraint.  The  new  cost
function can be further expressed as
Φ
1
(v) =
1
2
(
v
T
Dv+v
T
D
T
v
)
=
1
2
v
T
(
D+D
T
)
v
and  the unit  norm  vector which  maximise it,  is  therefore the
singular vector which corresponds to the largest singular value
ofD+D
T
.
Proof of Theorem 5:The property thatSscatters energy
from  each  node  equally  among  all  other  nodes  requires  that
all off-diagonal elements are identical, and thus thatAhas the
form
A=




a
1
a
0
···a
0
a
0
a
2
···a
0
··· ··· ··· ···
a
0
a
0
···a
K




.(43)
Orthogonality ofArequires thata
0
,...,a
K
satisfy
{
a
2
i
+ (K−1)a
2
0
= 1i= 1,...,K
a
0
(a
i
+a
j
) + (K−2)a
2
0
= 0∀i6=j
.(44)
The first constraint can be written asa
i
=±
√
1−(K−1)a
2
0
,
which  implies  that  all  diagonal  elements  are  identical  in
magnitude. The second constraint implies that ifa
0
6= 0(the
solutiona
0
= 0is ignored since it does not scatter energy), the
diagonal elements have also the same sign. Solving (44) with
a
1
=···=a
K
yieldsa
1
=±(2−K)/Kanda
0
=±2/K,
thus proving the theorem.