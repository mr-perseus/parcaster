# ParCaster

## Ausgangslage

Die Suche nach einem Parkplatz hat viele negative Auswirkungen, wie z. B. Zeit-, Treibstoff- und Emissionsverschwendung,
überhöhte Parkgebühren usw. Um die Parkplatzsuche zu erleichtern, ist das Ziel dieses Projekts durch die Anwendung von
maschinellem Lernen die Verfügbarkeit von Parkplätzen in der Stadt St. Gallen zur geschätzten Ankunftszeit der Nutzer
vorherzusagen. Der Nutzer erhält eine personalisierte Parkplatzempfehlung.

## Beschreibung ParCaster

ParCaster ist eine künstliche Intelligenz, welche freie Parkplätze in der Stadt St. Gallen vorhersagt. Das
zugrundeliegende Modell erstellt aufgrund von Inputs wie z.B. der Wetterprognose, von Feiertagen und Veranstaltungen
oder auch Schulferien entsprechende Prognosen.
Der Nutzer kann auf der Website die voraussichtliche Parkzeit eingeben, worauf ParCaster die Anzahl freie Parkplätze je
Parkgarage in der Stadt St. Gallen vorhersagt. Der Nutzer spart Zeit und Geld.

### Website ParCaster

https://parcaster.github.io/

## Vorgehen und Implementierung

Die Entwicklung des ParCasters kann in die Schritte Data-preprocessing, Modellentwicklung und Erstellung des
Web-Services unterteilt werden. Die einzelnen Schritte werden nachfolgend kurz erläutert:

### Data und Datapreprocessing

Die Daten, welche dem entwickelten Modell zugrunde liegen, werden vom Open-Data-Portal der Stadt St.
Gallen (https://daten.stadt.sg.ch/explore/dataset/freie-parkplatze-in-der-stadt-stgallen-pls/table/?disjunctive.phid&disjunctive.phname)
öffentlich zur Verfügung gestellt. In den Daten wird dargestellt, wie viele Parkplätze zu einem bestimmten Zeitpunkt in
den Parkgaragen der Stadt St. Gallen frei sind. Die zur Verfügung stehenden Daten sind aus dem Zeitraum zwischen Oktober
2019 und November 2023.
Bevor die Daten für die Kalibrierung des Modells verwendet werden, werden sie bereinigt. U.a. werden Ausreisser
entfernt. Zudem werden die Daten normalisiert und neue Features generiert.

### Modellentwicklung

Mit den vorbereiteten Daten wird ein LSTM (https://de.wikipedia.org/wiki/Long_short-term_memory) in Python trainiert.
Dabei werden unterschiedliche Werte für diverse Parameter (u.a. Anzahl Layer, Batch-Size, Optimizer) ausprobiert. Als
Loss-Function wird der MSE verwendet. Die Resultate werden mit Weigths & Bias getrackt. Die Ergebnisse können auf der
folgenden Homepage eingesehen werden: https://wandb.ai/parcaster/pp-sg-lstm/sweeps/zx34brsw

Das entsprechende Jupyter Notebook ist unter folgenden Link zu
finden: https://github.com/parcaster/parcaster/blob/master/model/W%26B_PPSG_LSTM.ipynb

Die untenstehenden Grafiken zeigen beispielhaft den Vergleich der Vorhersage mit den effektiven Werten nach dem
Training. Dabei sind die Vorhersagen für Testdaten gemacht worden, welche das Modell während des Trainings nicht
gesehen hat:

**Vorhersage vs. effektive Werte für den 01-03-2023 00:11 Uhr**

![Prediction 1](/docs/images/prediction-run-2cg5mebb-2023-03-01_0011.png "Prediction 01-03-2023 00:11")

**Vorhersage vs. effektive Werte für den 01-03-2023 02:27 Uhr**

![Prediction 2](/docs/images/prediction-run-2cg5mebb-2023-03-01_0227.png "Prediction 01-03-2023 02:27")

### Erstellung Web-Service

Das trainierte Modell wird über eine API auf Heroku deployed. Damit der Nutzer für den benötigten Zeitpunkt eine
Prognose zu den freien Parkplätzen abfragen kann, wird ihm ein User Interface zur Verfügung gestellt. Das Userinterface
wird mit Github Pages erstellt.

## Nächste Schritte

- Verbesserung des Modells (LSTM)
- Weitere Modelle testen (statistische Modelle, Transformer)
- Preprocessing der Daten verfeinern (z.B. weitere Features entwickeln)

## Mögliche Weiterentwicklungen

- Vorteile der einzelnen Parkmöglichkeiten in User Interface darstellen (z.B. überdacht vs. nicht-überdacht)
- Distanz zum Zielort in Vorschlag einarbeiten
- Parkplatz über User Interface vorreservieren für gewünschten Zeitpunkt

