.RP
.TL
Python Pakket en GUI voor FlowSOM
.AU
Elias Nijs
.AI
Computationele Biologie
Universiteit Gent
.AB
Dit verslag presenteert de ontwikkeling en implementatie van het
FlowSOM package voor python en de ontwikkeling van een grafische
gebruikersinterface (GUI) ervoor. Op deze manier maken we het FlowSOM algoritme
toegangkelijk voor zowel mensen met als zonder technische achtergrond.
.AE
.DA
.AM
.2C
.R1
accumulate
database bib.ref
.R2
.EQ
delim $$
.EN
.NH 1
Introductie
.LP
Flow cytometry is een techniek die in toenemende mate wordt gebruikt in
biomedisch onderzoek. Deze techniek maakt het mogelijk om de
fysische en chemische kenmerken van individuele cellen in een stroom
van vloeistof te meten, wat tot diepgaande inzichten in het functioneren
van cellen kan leiden. Een van de grootste uitdagingen van flow
cytometry is echter de analyse van de geproduceerde data, die vaak
uit duizenden tot miljoenen cellen bestaat, elk met meerdere gemeten
parameters. Het is een hoogdimensionale dataset, en traditionele
analysetechnieken zijn vaak ontoereikend.

In deze context is het FlowSOM-algoritme ontwikkeld. FlowSOM maakt gebruik
van Self-Organizing Maps (SOM) en meta clustering om de data op een
te visualiseren en interpretatie mogelijk te maken. Het heeft zich
bewezen als een krachtig hulpmiddel voor flow cytometry data-analyse.
Het blijft echter een uitdaging voor onderzoekers met een minder technische
achtergrond in computer wetenschappen om dit algoritme te gebruiken. Daarnaast
gebeurt computationele biologie steeds vaker in python.

Het doel van dit werk is om zowel een python pakket als grafische
gebruikersinterface (GUI) voor het FlowSOM-algoritme te implementeren. Op deze
manier kan de toegankelijkheid en de gebruikerservaring van dit algoritme
aanzienlijk verbeteren. Daarnaast kan het de drempel verlagen voor onderzoekers
om het FlowSOM-algoritme te gebruiken.

In dit verslag bespreken we onze aanpak voor het implementeren van de
GUI, de technische uitdagingen die we hebben overwonnen, en de snelheid en
efficientie die we hebben bereikt. Tot slot bespreken we ook mogelijke
uitbreidingen voor het pakket en de GUI.
.NH 1
Methoden
.LP
De methodologie van deze studie valt uiteen in twee componenten:
de implementatie van het FlowSOM python pakket en de implementatie van de
bijbehorende grafische gebruikersinterface (GUI).
.NH 2
FlowSOM Implementatie
.LP
De kern van dit onderzoek bestaat uit de implementatie van het
FlowSOM-algoritme. Deze implementatie is opgebouwd rond een modulaire structuur
terug te vinden in vier bestanden:
.I flowsom.FlowSOM,
.I flowsom.fs_dataclasses,
.I flowsom.fs_plotting,
.I flowsom.fs_reporting.
waarvan
.I flowsom.FlowSOM
en de paramterclasses uit
.I fs_dataclasses
publiek gedeclareerd worden in het pakket.

De FlowSOM implementatie wordt beschikbaar gesteld in een klasse genaamd FlowSOM, een
uitbreiding van de
.I BaseEstimator
en
.I ClusterMixin
klassen. Deze klasse bevat een tal aan functies voor het trainen, voorspellen en
visualiseren van de data, alsook een functie om een verslag te genereren.
Daarnaast houdt het object ook de relevante data zoals de parameters en de
tussenresultaten van het trainen bij.

De hoofd functie van het FlowSOM object is de
.I fit
methode. Dit is de methode die een model zal trainen. Dit trainen bestaat uit
drie delen: trainen van de som, bouwen van de minimum spanning tree en de
metaclustering. Het resultaat van elk van deze delen wordt ook intern opgeslaan
in het object en kan de gebruiker dus later nog verder gebruiken.

We raden aan het
Flowsom object op volgende manier te gebruiken:
.BD
.CW
.ps -1
model = FlowSOM(
        som_param,
        mst_param,
        hcc_param).fit(data)
.ps +1
.DE
We zien hierbij twee zaken: de initialisatie van het object en het oproepen van
de
.I fit
methode.
De argumenten van de initialisatie zijn dataclasses die de parameters van elk
deel van het algoritme bevatten. Er werd gekozen om deze in een dataclass te
steken om de paramters overzichtelijk te structureren. De parameters van de fit
functie zijn een pandas dataframe die de cytometrie data bevat en een optie om
logging aan te zetten met
.CW verbose=True .

De functionaliteit van de FlowSOM-klasse strekt zich verder uit tot
data- en visualisatiefuncties waaronder functies om de minimum spanning tree met
of zonder metalcusters te plotten, een functie om het som grid met of zonder
metaclusters te plotten, ... Meer informatie omtrent deze is terug te vinden in de documentatie.

Het FlowSOM-pakket omvat ook een uitgebreide rapport functie, deze is
in staat is een verslag van de analyse te genereren in
markdown syntax. Dit rapport bevat een samenvatting van de
FlowSOM-analyseresultaten waaronder de model parameters, visualisaties en
numerisch resultaten voor elke node en elke metacluster. Er werd gekozen voor
markdown zodat de gebruiker kan kiezen in welke stijl hij het verslag wilt zien.
.NH 2
FlowSOMTool: De GUI
.LP
De tweede component van onze methodologie betreft de ontwikkeling van
FlowSOMTool, een GUI ontworpen aan de hand van python bindings voor de
ImGui-bibliotheek. Deze interface streeft
ernaar de interactie met het FlowSOM-algoritme zo eenvoudig mogelijk te maken
zodat mensen met minder technische achtergrond ook het algoritme kunnen
gebruiken.

FlowSOMTool biedt gebruikers een breed scala aan mogelijkheden, te beginnen met
de selectie van de FlowSOM-bestanden die geanalyseerd moeten worden. Vervolgens
stelt het de gebruikers in staat om de parameters voor het algoritme aan te
passen, het algoritme uit te voeren, en de resultaten te exporteren voor verdere
analyse. Een cruciaal onderdeel van de GUI is het vermogen om de gegenereerde
plots te visualiseren. Hierdoor krijgen gebruikers de mogelijkheid om de
FlowSOM-analyse op een visueel aantrekkelijke manier te inspecteren. Deze GUI is
een prototype en is bedoeld als platform waar meer geavanceerdere interactie
mogelijkheden kunnen op verder gebouwd worden.

Deze methodologie, met zijn twee componenten, zorgt voor een
veelzijdige mogelijkheden voor de analyse van flowcytometriegegevens, terwijl het
tegelijkertijd de flexibiliteit en aanpassingsvermogen behoudt die nodig zijn
voor individueel onderzoek.

.NH 1
Resultaten

.NH 1
Conclusie

.NH 1
Toekomstig werk
.LP
Hoewel de huidige versie van onze FlowSOM-tool al krachtige visualisatie- en
analysefuncties biedt, zetten we ons in voor een voortdurende evolutie van de
tool, gericht op het verhogen van de bruikbaarheid en het uitbreiden van de
functionaliteit. We identificeren een aantal mogelijke verbeteringen die voor
toekomstige implementatie worden overwogen:
.NH 2
Highlighting en Tooltip-functie
.LP
Door een functie toe te voegen die knooppunten markeert wanneer een gebruiker
met de muis over het knooppunt beweegt, wordt de interactie tussen de gebruiker
en de visualisaties verder versterkt. Dit verbetert niet alleen de leesbaarheid,
maar zorgt ook voor een uitgebreid begrip van de onderliggende gegevens. Het
combineren van deze functie met een tooltip kan meer gedetailleerde informatie
over het geselecteerde knooppunt weergeven, wat de gebruiker toegang geeft tot
alle individuele dimensies van de gegevens.
.NH 2
Filtermogelijkheden
.LP
Een andere geplande verbetering is het geven van meer controle aan de gebruikers
over welke knooppunten worden weergegeven. Dit kan worden bereikt door
filterregelaars toe te voegen die gebaseerd zijn op verschillende criteria zoals
clusterlidmaatschap, waardebereik, enz.
.NH 2
Kleurenkaart aanpassing
.LP
De mogelijkheid voor gebruikers om de kleurenkaart van de gegevens te wijzigen
kan een waardevolle toevoeging zijn. Deze functionaliteit kan helpen om de
knooppunten visueel te differentiëren op basis van verschillende dimensies of
metrieken.
.NH 2
3D-visualisatie
.LP
Voor bijzonder complexe, hoog-dimensionale gegevens kan de toevoeging van
3D-visualisatie, inclusief de mogelijkheid om de 3D-weergave te roteren,
bijzonder waardevol zijn.
.NH 2
Details op aanvraag
.LP
We plannen om een optie toe te voegen waarbij gebruikers op een knooppunt kunnen
klikken om een meer gedetailleerd beeld te krijgen in een apart paneel. Dit
paneel kan alle dimensies en andere details van de gegevens bevatten.
.NH 2
Vergelijkingsmodus
.LP
Een functie die gebruikers in staat stelt om meerdere knooppunten of datapunten
te selecteren en een vergelijking van hun gegevens weer te geven, kan bijzonder
nuttig zijn bij het identificeren van overeenkomsten of verschillen in
hoog-dimensionale gegevens.
.NH 2
Zoekfunctie
.LP
Tot slot overwegen we de toevoeging van een zoekfunctie waarmee gebruikers
specifieke knooppunten of datapunten kunnen zoeken op basis van verschillende
parameters.

.LP
Deze voorgenomen verbeteringen weerspiegelen onze inzet om de FlowSOM-tool nog
gebruiksvriendelijker en krachtiger te maken. We zijn enthousiast over de
voortzetting van dit project en de potentiële waarde die deze toekomstige
verbeteringen zullen bieden voor de gebruikersgemeenschap.

.NH 1
Conclusies
.LP
Onze FlowSOM-tool is een veelzijdig en krachtig hulpmiddel voor het
visualiseren en analyseren van complexe, hoogdimensionale gegevens. Door
gebruik te maken van zelforganiserende kaarten en minimum spannende bomen,
biedt FlowSOM een effectieve methode voor clustering en visualisatie, waardoor
gebruikers bruikbare inzichten kunnen halen uit hun gegevens.

Hoewel de huidige versie van de tool al aanzienlijke waarde biedt, erkennen we
dat er altijd ruimte is voor verbetering en uitbreiding. De geplande
verbeteringen die we hebben geïdentificeerd - zoals het toevoegen van functies
voor markering, tooltip, filtering, aanpassing van kleurenkaarten,
3D-visualisatie, gedetailleerde weergave op aanvraag, tijdreeksanimatie,
vergelijkingsmodus en zoeken - beloven de bruikbaarheid en functionaliteit van
de tool aanzienlijk te verhogen.



.NH 1
Referenties
