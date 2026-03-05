#import "../template.typ": *

= Gans

Si tratta di un generatore per dati sintetici, che impara a generare dati simili a quelli di un dataset di addestramento.

L'idea è avere due reti neurali che competono tra loro: una rete generatrice (Generator) che cerca di creare dati sintetici realistici, e una rete discriminatrice (Discriminator) che cerca di distinguere tra dati reali e dati sintetici.

Non abbiamo versimiglianza, si usa il gioco tra generatore e disciminaore. Dati dati reali ovvero una distribuzione $P_("data")$ facciamo training del modello sul dataset reale. L'output vogliamo sollecitare il modello a generare dati sintetici che siano simili a quelli reali, ovvero che seguano la stessa distribuzion. La distribuzione del modello vogliamo che sia simile a quella reale, ovvero:
$
  P_("model") tilde P_("data")
$

Il discriminatore e il generatore sono due modelli che vengono allenati assime. Se il generatore inganna sempre il discriminatore ha capito bene la distribuzione reale dei dati.

Generatore $G(z)$
- input: vettore rumoroso $z tilde p_z(z)$
- output: dato sintetico $G(z)$

Discriminatore $D(x)$
- input: dato $x$ (reale o sintetico)
- output: probabilità che $x$ sia reale, appartiene a $D(x) in [0,1]$
- il compito è distinguere se l'ingresso $x$ è vero o falso

== generatore

Il generarore è una NN con parametro $w$ che definsice la mappa $g: Z->X$ è la mappatura che impara. Questa mappatura realizza il fitting dei dati:
$
  x = G_w(z) "con" x in {x_n}_(n=1)^N
$
Poichè risulta difficile ottimizare la funzione likehood si usano le gan:
- Discriminatore (1 real, 0 fake)

== Discriminatore

Il discriminatore è una NN con parametro $theta$ che definisce la mappa $d: X-> [0,1]$ che stima la probabilità che un dato $x$ sia reale. Il discriminatore viene addestrato a distinguere tra dati reali e sintetici, cercando di massimizzare la probabilità di classificare correttamente i dati reali e minimizzare la probabilità di classificare erroneamente i dati sintetici come reali.

Viene utilizzata la cross-entroy per ottimizzare il discrimatore. L'idea è usare la cross-entropy per misurare la distanza tra la distribuzione reale dei dati e la distribuzione generata dal modello. Il discriminatore cerca di massimizzare la probabilità di classificare correttamente i dati reali e minimizzare la probabilità di classificare erroneamente i dati sintetici come reali.

//AGGIUNGERE FORMULE

Se poniamo $t=1$ per fati veri e $t=0$ per dati sintetici, la cross-entropy è data da:
$
  L(theta) = -E_(x~P_("data"))[log D_theta(x)] - E_(z~P_z(z))[log(1 - D_theta(G_w(z)))]
$

Modalità adversirial, la loss viene minimizzata rispetto a $theta$ ovvero rispetto al discriminatore. Vogliamo un discriminatore che classifichi. Vogliamo qundi che la prima parte della formula $log D_theta(x_n)$ sia massimizata, ovvero che il discriminatore assegni una probabilità alta ai dati reali. Vogliamo anche che la seconda parte della formula $log(1 - D_theta(G_w(z)))$ sia massimizzata, ovvero che il discriminatore assegni una probabilità bassa ai dati sintetici generati dal generatore.

Se il generatotre arriva alla solizione perfetta è che non distingue più numlla (1/2 e 1/2 nei due casi). In questo caso il discriminatore non riesce più a distinguere tra dati reali e sintetici, e la distribuzione generata dal modello è simile a quella reale:
$
  P(D_theta(x) = 1) = 0.5
$
L'obbiettivo della Loss è portare a somma 0, alla condizione in cui il discriminatore non distingue più.

Generazione: dopo il training (se buono) abbiamo un generatore che è in grado di generare dati sintetici. L'idea è portare la distribuzione iniziale del generatore verso quella reale

== Training su MINST

Viene utilizzata la leaky Relu. Si tratta di una variante della funzione di attivazione ReLU che consente un piccolo gradiente anche per valori negativi, evitando il problema dei neuroni "morti" che possono verificarsi con la ReLU standard. La leaky ReLU è definita come:
$
  f(x) = cases(
    x & "se" x > 0,
    alpha x & "se" x <= 0
  )
$
Bisgona quindi mettere un parametro $0$.

Il discriminatore è una funzione motlo semplice lineare. é necessario applicare una sigmoide finale per ottenere uan probabilità. I dropout prevedono l'over fitting ecc.

Solitamente l'immagine di input (se si ha un immagine) viene riscalata tra [-1,1] per facilitare l'apprendimento del modello. Il generatore invece produce output tra [-1,1] e quindi è necessario scalare i dati di input in questo intervallo.

=== Esempio //aggiungere

== CycleGAN

Si tratta di una variante delle GAN che consente di eseguire la traduzione di immagini da un dominio a un altro senza la necessità di avere coppie di immagini corrispondenti. Ad esempio, è possibile utilizzare CycleGAN per trasformare foto di cavalli in foto di zebre, o viceversa, senza dover avere coppie di immagini corrispondenti.


CycleGAN utilizza due generatori e due discriminatori. Un generatore (G) viene addestrato a trasformare immagini da un dominio A a un dominio B, mentre l'altro generatore (F) viene addestrato a trasformare immagini da un dominio B a un dominio A. I discriminatori (D_A e D_B) vengono addestrati a distinguere tra immagini reali e immagini generate nei rispettivi domini.


Andiamo quindi a generare una mappa tra i due spazi di immagini, ovvero tra il dominio A e il dominio B. Il generatore G impara a trasformare immagini da A a B, mentre il generatore F impara a trasformare immagini da B a A. I discriminatori D_A e D_B vengono addestrati a distinguere tra immagini reali e immagini generate nei rispettivi domini.

Il ciclo di addestramento di CycleGAN prevede due fasi principali: la fase di addestramento dei generatori e la fase di addestramento dei discriminatori. Durante la fase di addestramento dei generatori, i generatori vengono addestrati a generare immagini realistiche nei rispettivi domini, cercando di ingannare i discriminatori. Durante la fase di addestramento dei discriminatori, i discriminatori vengono addestrati a distinguere tra immagini reali e immagini generate nei rispettivi domini.

//aggiungere pdf e codice training

== Interpretazione geometrica

Non abbiamo lo spazio latente, ma abbiam due Mainfold
$
  M_a = M_b
$
Il generatore impara a mappare un punto da un manifold all'altro. Il discriminatore invece impara a distinguere tra i due manifold, cercando di identificare le differenze tra di essi. L'obbiettivo è portare i due manifold a sovrapporsi, in modo che il discriminatore non riesca più a distinguere tra di essi.


