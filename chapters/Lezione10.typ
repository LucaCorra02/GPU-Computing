#import "../template.typ": *

= PyTorch autograd

== Fondamenti Matematici del Gradiente

=== Definizione di Gradiente

Il *gradiente* è un operatore differenziale che si applica a funzioni scalari definite su spazi vettoriali. Data una funzione:
$
  f: R^D -> R
$
che mappa un vettore di $D$ dimensioni in uno scalare, il gradiente di $f$ è definito come:
$
  nabla f: R^D -> R^D
$

Il gradiente produce un *vettore* che contiene tutte le *derivate parziali* della funzione rispetto a ciascuna variabile:
$
  nabla f(x) = vec((partial f)/(partial x_1), (partial f)/(partial x_2), dots.v, (partial f)/(partial x_D))
$

#nota()[
  Il gradiente trasforma una funzione scalare in un *campo vettoriale*: ad ogni punto $x$ dello spazio viene associato un vettore $nabla f(x)$.
  
  Nel contesto del deep learning:
  - $f$ rappresenta la *funzione di loss*
  - $x$ rappresenta i *parametri* del modello (pesi e bias)
  - $nabla f(x)$ indica come modificare i parametri per ridurre la loss
]

=== Derivate Parziali

Una *derivata parziale* misura il tasso di variazione di una funzione rispetto a *una singola variabile*, mantenendo costanti tutte le altre.

Data $f(x_1, x_2, dots, x_D)$, la derivata parziale rispetto a $x_i$ è:
$
  (partial f)/(partial x_i) = lim_(h -> 0) (f(x_1, dots, x_i + h, dots, x_D) - f(x_1, dots, x_i, dots, x_D))/h
$

#esempio()[
  Consideriamo la funzione quadratica in due variabili:
  $
    f(x_1, x_2) = x_1^2 + 3x_1 x_2 + 2x_2^2
  $

  *Derivata parziale rispetto a $x_1$*:

  Trattiamo $x_2$ come una costante:
  $
    (partial f)/(partial x_1) = 2x_1 + 3x_2
  $

  *Derivata parziale rispetto a $x_2$*:

  Trattiamo $x_1$ come una costante:
  $
    (partial f)/(partial x_2) = 3x_1 + 4x_2
  $

  *Gradiente completo*:
  $
    nabla f(x_1, x_2) = vec(2x_1 + 3x_2, 3x_1 + 4x_2)
  $

  In un punto specifico, ad esempio $(x_1, x_2) = (1, 2)$:
  $
    nabla f(1, 2) = vec(2(1) + 3(2), 3(1) + 4(2)) = vec(8, 11)
  $
]

=== Interpretazione Geometrica del Gradiente

Il gradiente ha diverse interpretazioni geometriche fondamentali:

*1. Direzione di massima crescita*

Il vettore $nabla f(x_0)$ punta nella direzione in cui la funzione $f$ *cresce più rapidamente* a partire dal punto $x_0$.

*2. Vettore normale alle curve di livello*

Il gradiente in un punto è *perpendicolare* alla curva (o superficie) di livello passante per quel punto.

*3. Magnitudine come tasso di crescita*

Il modulo $||nabla f(x_0)||$ rappresenta il *tasso di crescita massimo* di $f$ in $x_0$.

#esempio()[
  Consideriamo il paraboloide:
  $
    f(x, y) = x^2 + y^2
  $

  Il gradiente è:
  $
    nabla f(x, y) = vec(2x, 2y)
  $

  #figure(
    cetz.canvas({
      import cetz.draw: *

      // Configurazione
      let scale = 1.5
      
      // Assi
      line((-3, 0), (3, 0), mark: (end: ">"))
      content((3.3, 0), $x$)
      line((0, -3), (0, 3), mark: (end: ">"))
      content((0, 3.3), $y$)

      // Curve di livello (cerchi concentrici)
      for r in (0.5, 1.0, 1.5, 2.0, 2.5) {
        circle((0, 0), radius: r, stroke: (paint: blue.lighten(30%), thickness: 1pt))
      }

      // Etichette curve di livello
      content((0.5, 0), text(size: 7pt, $c_1$), anchor: "south", fill: blue)
      content((1.5, 0), text(size: 7pt, $c_2$), anchor: "south", fill: blue)
      content((2.5, 0), text(size: 7pt, $c_3$), anchor: "south", fill: blue)

      // Punti campione
      let points = ((1, 0.5), (0.7, 1.5), (-1, 1), (-1.5, -0.5), (0.5, -1.5))
      
      for pt in points {
        let (x, y) = pt
        // Punto
        circle((x, y), radius: 0.08, fill: red, stroke: none)
        
        // Calcola gradiente: ∇f = (2x, 2y)
        let grad_x = 2 * x * 0.3  // Scala per visualizzazione
        let grad_y = 2 * y * 0.3
        
        // Disegna vettore gradiente
        line((x, y), (x + grad_x, y + grad_y), 
             mark: (end: ">"), 
             stroke: (paint: red, thickness: 1.5pt))
      }

      // Origine
      circle((0, 0), radius: 0.1, fill: green, stroke: none)
      content((0, -0.4), text(size: 8pt, $(0,0)$), fill: green)

      // Legenda
      rect((-2.8, 2), (-1.2, 2.8), fill: white.transparentize(20%), stroke: 0.5pt)
      line((-2.6, 2.6), (-2.2, 2.6), stroke: (paint: blue.lighten(30%), thickness: 1pt))
      content((-1.9, 2.6), text(size: 7pt, [Curve di livello]), anchor: "west")
      line((-2.6, 2.3), (-2.2, 2.3), mark: (end: ">"), stroke: (paint: red, thickness: 1.5pt))
      content((-1.9, 2.3), text(size: 7pt, [$nabla f$]), anchor: "west")
    }),
    caption: [Campo vettoriale del gradiente $nabla f(x,y) = vec(2x, 2y)$ per il paraboloide $f(x,y) = x^2 + y^2$. I vettori rossi (gradienti) sono perpendicolari alle curve di livello blu e puntano verso l'esterno (crescita).]
  )

  *Osservazioni*:
  - All'origine $(0, 0)$, il gradiente è nullo: $nabla f(0,0) = vec(0,0)$ (punto di minimo)
  - Nei punti lontani dall'origine, i vettori gradiente sono più lunghi (crescita più rapida)
  - I gradienti puntano *radialmente verso l'esterno* (direzione di massima crescita)
  - Sono tutti *perpendicolari* ai cerchi (curve di livello)
]

=== Curve di Livello (Level Sets)

Le *curve di livello* (o superfici di livello in dimensioni superiori) sono l'insieme dei punti dove la funzione assume un *valore costante*:
$
  cal(L)_c = {x in R^D : f(x) = c}
$

#nota()[
  *Proprietà fondamentale*: Il gradiente $nabla f(x_0)$ in un punto $x_0$ è *ortogonale* alla curva di livello passante per $x_0$.
  
  Questo significa che il gradiente punta nella direzione che *esce* dalla curva di livello, verso valori crescenti di $f$.
]

#esempio()[
  Per la funzione $f(x, y) = x^2 + 2y^2$, le curve di livello sono ellissi:
  $
    x^2 + 2y^2 = c
  $

  #figure(
    cetz.canvas({
      import cetz.draw: *

      // Assi
      line((-3, 0), (3, 0), mark: (end: ">"))
      content((3.3, 0), $x$)
      line((0, -2.5), (0, 2.5), mark: (end: ">"))
      content((0, 2.8), $y$)

      // Curve di livello (ellissi)
      for c in (0.5, 1.0, 2.0, 3.0) {
        let a = calc.sqrt(c)
        let b = calc.sqrt(c / 2)
        
        // Disegna ellisse parametrica
        let points = ()
        for i in range(0, 101) {
          let t = i * 2 * calc.pi / 100
          points.push((a * calc.cos(t), b * calc.sin(t)))
        }
        line(..points, stroke: (paint: blue.lighten(40%), thickness: 1pt), close: true)
      }

      // Etichette
      content((calc.sqrt(0.5), 0.1), text(size: 7pt, $c=0.5$), anchor: "south", fill: blue)
      content((calc.sqrt(2), 0.1), text(size: 7pt, $c=2$), anchor: "south", fill: blue)

      // Punti e gradienti
      let test_points = ((1, 0.5), (-1, 0.7), (0.5, -1))
      
      for pt in test_points {
        let (x, y) = pt
        circle((x, y), radius: 0.08, fill: red, stroke: none)
        
        // Gradiente: ∇f = (2x, 4y)
        let grad_x = 2 * x * 0.25
        let grad_y = 4 * y * 0.25
        
        line((x, y), (x + grad_x, y + grad_y),
             mark: (end: ">"),
             stroke: (paint: red, thickness: 1.5pt))
      }

      // Origine
      circle((0, 0), radius: 0.1, fill: green, stroke: none)
    }),
    caption: [Curve di livello ellittiche per $f(x,y) = x^2 + 2y^2$. Il gradiente $nabla f = vec(2x, 4y)$ è perpendicolare alle ellissi.]
  )
]

#attenzione()[
  Le curve di livello sono fondamentali nell'ottimizzazione:
  
  - Un *passo di gradient descent* ci sposta da una curva di livello a un'altra con valore inferiore
  - La direzione *perpendicolare* alla curva (il gradiente) è la direzione di *massimo cambiamento*
  - Seguire il gradiente negativo ci porta verso valori decrescenti di $f$ (verso il minimo)
]

=== Teorema della Direzione di Massima Discesa

Il teorema fondamentale che giustifica l'uso del gradiente nell'ottimizzazione:

*Teorema*: Dato un punto $x_0 in R^D$ e una funzione differenziabile $f: R^D -> R$, la direzione di *massima discesa* (che minimizza $f$ localmente) è data da:
$
  d^* = -nabla f(x_0)
$

Ovvero, il *negativo del gradiente* indica la direzione in cui la funzione decresce più rapidamente.

==== Dimostrazione tramite Sviluppo di Taylor

Consideriamo lo sviluppo di Taylor al primo ordine di $f$ intorno a $x_0$:
$
  f(x_0 + alpha d) approx f(x_0) + alpha nabla f(x_0)^T d
$

dove:
- $d$ è una direzione unitaria ($||d|| = 1$)
- $alpha > 0$ è uno step size piccolo

#nota()[
  Lo sviluppo di Taylor ci dice che per piccoli spostamenti $alpha d$ da $x_0$, la funzione cambia approssimativamente di:
  $
    Delta f approx alpha nabla f(x_0)^T d
  $
]

Vogliamo trovare la direzione $d^*$ che *minimizza* $f(x_0 + alpha d)$, ovvero che rende $Delta f$ il più negativo possibile:
$
  d^* = arg min_(||d||=1) nabla f(x_0)^T d
$

Per il *teorema di Cauchy-Schwarz*:
$
  nabla f(x_0)^T d <= ||nabla f(x_0)|| dot ||d|| = ||nabla f(x_0)||
$

L'uguaglianza si ottiene quando $d$ è *allineato* con $nabla f(x_0)$. Il minimo si ottiene quando:
$
  d^* = -(nabla f(x_0))/(||nabla f(x_0)||)
$

Ovvero, la direzione *opposta* al gradiente (normalizzato).

#nota()[
  *Conclusione*: Muoversi nella direzione $-nabla f(x_0)$ garantisce la *massima riduzione* della funzione $f$ in un intorno di $x_0$.
  
  Questo è il principio alla base del *Gradient Descent*!
]

=== Gradient Descent: Algoritmo di Base

L'algoritmo di *gradient descent* per minimizzare una funzione $f(x)$ è:

$
  x_(k+1) = x_k - alpha_k nabla f(x_k)
$

dove:
- $x_k$: valore dei parametri all'iterazione $k$
- $alpha_k > 0$: *learning rate* (passo di discesa)
- $nabla f(x_k)$: gradiente calcolato in $x_k$

*Algoritmo iterativo*:
1. Inizializza $x_0$ (casualmente o con euristica)
2. Per $k = 0, 1, 2, dots$ fino a convergenza:
   - Calcola il gradiente: $g_k = nabla f(x_k)$
   - Aggiorna i parametri: $x_(k+1) = x_k - alpha_k g_k$
   - Controlla convergenza: se $||g_k|| < epsilon$, termina

#nota()[
  *Criteri di convergenza*:
  
  - *Condizione ideale*: $nabla f(x_k) = bold(0)$ (punto stazionario)
  - *Condizione pratica*: $||nabla f(x_k)|| < epsilon$ per una soglia piccola $epsilon > 0$
  
  #attenzione()[
    Un gradiente nullo può indicare:
    - Un *minimo locale* (desiderato)
    - Un *massimo locale* (indesiderato)
    - Un *punto di sella* (né minimo né massimo)
    
    In deep learning, i *punti di sella* sono molto più comuni dei minimi locali in alta dimensione.
  ]
]

#esempio()[
  Minimizziamo $f(x, y) = x^2 + 4y^2$ partendo da $(x_0, y_0) = (3, 2)$.

  *Gradiente*:
  $
    nabla f(x, y) = vec(2x, 8y)
  $

  *Iterazioni con $alpha = 0.1$*:

  - $k=0$: 
    - $x_0 = vec(3, 2)$
    - $nabla f(x_0) = vec(6, 16)$
    - $x_1 = vec(3, 2) - 0.1 vec(6, 16) = vec(2.4, 0.4)$
  
  - $k=1$:
    - $nabla f(x_1) = vec(4.8, 3.2)$
    - $x_2 = vec(2.4, 0.4) - 0.1 vec(4.8, 3.2) = vec(1.92, 0.08)$
  
  - $k=2$:
    - $nabla f(x_2) = vec(3.84, 0.64)$
    - $x_3 = vec(1.92, 0.08) - 0.1 vec(3.84, 0.64) = vec(1.536, 0.016)$

  Convergenza verso il minimo $(0, 0)$ dove $f(0,0) = 0$.
]

#attenzione()[
  *Scelta del Learning Rate $alpha$*:
  
  - $alpha$ troppo *piccolo*: convergenza molto lenta, molte iterazioni
  - $alpha$ troppo *grande*: oscillazioni, possibile divergenza
  - $alpha$ *ottimale*: bilanciamento tra velocità e stabilità
  
  In pratica si usano tecniche di *learning rate scheduling* (decay, adaptive methods come Adam).
]

=== Visualizzazione del Processo di Ottimizzazione

#figure()[
  #cetz.canvas({
    import cetz.draw: *

    // Assi
    line((-3, 0), (3, 0), mark: (end: ">"))
    content((3.3, 0), $x$)
    line((0, -0.5), (0, 3.5), mark: (end: ">"))
    content((0, 3.8), $f(x)$)

    // Funzione quadratica
    let points = ()
    for i in range(-30, 31) {
      let x = i / 10.0
      let y = x * x * 0.3 + 0.2
      points.push((x, y))
    }
    line(..points, stroke: (paint: blue, thickness: 2pt))

    // Traiettoria gradient descent
    let trajectory = (
      (2.2, 1.65),
      (1.5, 0.875),
      (0.9, 0.443),
      (0.5, 0.275),
      (0.2, 0.212),
      (0.05, 0.2075),
    )

    // Disegna traiettoria
    for i in range(trajectory.len() - 1) {
      let pt1 = trajectory.at(i)
      let pt2 = trajectory.at(i + 1)
      line(pt1, pt2, mark: (end: ">"), stroke: (paint: red, thickness: 1.5pt))
    }

    // Punti sulla traiettoria
    for (i, pt) in trajectory.enumerate() {
      circle(pt, radius: 0.08, fill: red, stroke: none)
      if i == 0 {
        content((pt.at(0), pt.at(1) + 0.3), text(size: 8pt, $x_0$), fill: red)
      } else if i == trajectory.len() - 1 {
        content((pt.at(0) - 0.3, pt.at(1)), text(size: 8pt, $x^*$), fill: red)
      }
    }

    // Minimo
    circle((0, 0.2), radius: 0.1, fill: green, stroke: (paint: green, thickness: 2pt))

    // Etichette
    content((1.5, 3), text(size: 10pt, $f(x) = x^2$), fill: blue)
    content((2, 2.3), text(size: 9pt, [Gradient Descent]), fill: red)
  })
  
  Visualizzazione del gradient descent su una funzione unidimensionale. La traiettoria rossa mostra i passi iterativi che convergono al minimo (punto verde).
]

== Cos'è un graidente

Partiamo da funzioni definite su uno spazio $D$ (input vettori) che mappano in un reale.

$gradient$ è un operatore che viene applicato a una funzioni $f$ (rappresnta una rete neurale):
$
  gradient f: R^D->R^D
$
#informalmente[
  un gradiente è un vettore di derivate parziali.
  Applichiamo la definizione di derivata solo che consideriamo il tasso di variazioni su una singola variabile, tenendo costanti le altri.
]
Si fa difierenziazione solo rispetto a una variabile ecc //aggiugnere formula

#esempio()[
  $
    f: R^2 -> R
  $
  Otteniamo un campo vettoriale:
  $
    delta f(x) = (alpha f) / (alpha x_1) hat(i)+ (alpha f) / (alpha x_2) hat(j)
  $
  Il grafico 3d ci dice che dato un vettore che parte in un punto ci dice il grado di pendenza di quella funzione. I vettori sono più o meno intesi (modulo del vettore). i vettori hanno direzioni pendenze e intesità diverse man mano che mi muovo nel dominio
]

Aggiungee latro esempio con due variabili.

=== Curve di livello

Sono importanti perchè quando usiamo il gradiente per raggiungere un punto di minimo in una funzione, usiamo le curve di livello. Un passo viene fatto da una cruva di livello ad un altra curva di livello (scegliendo una certa direzioni). Posso dstinguere le curve di livello in base a quelle che mi favoriscono la discesa verso il basso.

//aggiungere immagine campo vettoriale generato dal paraboloide

Il gradiente in ogni punto della funzione è perpendicolare alla tangente della curva di livello. Il gradiente viene interpretato come direzione

*Teo*: (direzione di massima discesa) //aggiungere
#informalmente()[
  Il segno serve altrimenti andre in salita
]
Lo sviluppo di taylor speiga il perchè:
$
  f(x) = f(x_0)+ gradient (f_x_0)^T dot (x-x_0)
$
#informalmente()[
  La funzione in un punto intorno da $x_0$. Se voglio usare la funzione di taylor per approssimare l'intorno di un punto devo ricorrere al gradiente moltiolicato (prodotto scalre)

  //riguardare
  Voglio valutare la funzione in x scritto come lo scostamento di x dal bho.\
  Vado a sostituire nella formula
  $
    x_i - alpha gradient(f_(x_o))
  $
]
Mi muovo verso un valore del dominio in cui la funzione ha valore minore. La migliore direzione è quella che coincide con $gradient f(X_0)$. Per il teorema se scelgo $d^*$ in questo modo ho una discesa ottimale in qualsiasi putno della funzione

=== algoritmo

//agiungere algoritmo
Il passo è $alpha_k$ (learning rate), potrebbe variari tutte le volte che ho un passo di iterazione tramite degli algoritmi

Criteri di stop. Vogliamo arrivare al minimo assoluto $gradient f(x_k) = 0$. Questa equazione è soddisfatta da un minimo relativo o minimi locali,oppure in caso di grafico a sella ho una variabile che mi fa salire e l'altra scendere.

Dobbiamo fissare una soglia $epsilon$. Se soo sotto la soglia mi fermo:

== Jacobiano
Quando abbiamo delle funzioni vettore->vettore sono delle funzioni non banali $R^n->R^m$ (layer dei modelli deep).

Lo jacobiano è la generalizzazione del gradiente per le matrici. Ogni componente della funzione $F(hat(x))$ è una funzione $f_x(overline(x)))$ a sua volta(composizoni di funzioni). Il gradiente viene esteso, deve differenziare rispetto alle funzioni e alle variabili:

La prima riga è la raccolta delle derivate parziali per la prima componente della funzione $F$. L'ultima riga è la derivata parziale rispetto all'ultima componente di $F$ grande.
#esempio()[
  Date $f_1 = x^2+y$ e $f_2 = x y$, ottengo $2x = (alpha f)/x$
]

#esempio()[
  Se prendo la funzione $h = g(overline(f))$. La f (composizoni di funzioni) è una variabile anche se deriva dal calcolo di $f_1(overline(x)), dots, f_k (overline(x))$.

  Se devo calcolare il gradiente di $h$ devo arrivare a moltiplicare lo jacobiao della funzione $f$ per la derivata priam di $g$:

  #esempio()[
    h è la composione di g e di f, posso dire che $h$ diepende da x, $h(overline(x))=g(f(x)))$.

    Posso usare la composizione in qualche modo //riguardare
  ]
]
=== regola della catena

Se predo due funzioni f e g e la loro composizione è derivabile ottendo $f'(g(x)) = r^k$

Posso fare la compsizione in quanto :$f(x_0)$ produce uan cosa di dim $m$ mentre $g$ prende $m$ e restituisce $k$:
$
  g(f(x_o)) -> z in R^k
$

#attenzione()[
  per calcolare i gradienti bisogna fare prodotti basati sugli jacobiani
]

== Loss

Una Loss :
$
  R^D -> R
$
è una funzione che mappa da $R^D$ a $R$ è l'ultimo step del procedimento visto fino adesso. L'obbiettivo è minimizzare questo funzione considerando il set di parameti $w$ come variabile.

Apllichiamo l'algoritmo di discesa dei gradienti per minimizzare la funzione

== Differenziazione automatica

Implementano il calcolo *esatto* (non approssimato) delle derivate di una funzione, si tratta di un meccanisomo di calcolo dei gradienti.

Vogliamo calcolare il gradiente della funzione loss avendo a disposizione un numero di elementi che mappano l'input sull'output (sono un numero elevato).

Potrei prendere ogni singolo elemento e farlo transistare nel mdoello -> calcolare la predizione -> caloclare l'errore -> modificare. Si tratta dell'approccio ideale. Tuttavia è meno robusto della discesa del gradiente stocastico e computazionalmente costoso

=== Graident stocastic

Scelgo a caso un elemento del dominio e sulal base dell'errore che si comette $L$, modifico i pesi. Un altro approccio più funzionante è basato su batch.

Lavora nel seguente modo:
$
  w_t+1 = w_t - n 1 / |B_t| sum_(n in B_t) gradient_w underbrace(l(f(x_n;w_t),y_n), "valore commesso medio")
$
Prendo un batch $B$ di tot elementi (piccola rispetto al dataset) e applico la variazione dei pesi sul valor medio di tutti i batch (non sul singolo esempalre)

Quello che succedo è che se prendo il gradiente medio calcolato sul batch è che non riesco ad avere la discesa ottimale in quel punto, tuttavia ho dei vantaggi :
- Molta velocità, meno computazioni
- Capacità di fare esapece dei minimi locali (in quanto è più rumoroso)

== Grafo computazionale

PyTorch implementa dietro le quinte autograd. Costuisce un grafo orientato (grafo diretto aicilico) che traccia tutte le operaizoni in tempo reale:
- I nodi sono le operaizoni ($+, -, *, "/"$)
- Gli archi sono l'interdipendenza tra questi nodi

Le aprime chiavi `requires_grad=True` associata ai tensori e l'operazione `backward()` ci permettono di fare il forward e modificare i pesi.

I gradienti vengono accumolati, ci sono un pool di gradienti che vengono utilizzati per modificare poi i pesi, vengono accumolati nel campo `.grad` di un tensore. La regola di Differenziazione viene applicata al grafo.

#esempio()[
  Regressione logistica (regressione per la classificazione).
  $D={x_i,y_i}$

  Se mi fermo al modello lineare ottengo un iperpiano, commette errori. Se prendo la funzione sigmoide o logistica ci permette di dire che se prendo una soglia (ad esempio $0.5$) ci permette di dire che se la rete emette $z$ (logits) da +/-inifnito la funzione di attivazione li riposta in un dominio più semplice, posso discriminarli in due classi 0 e 1.

  La funione di loos è la cross-entropy.

  Con $y=1$ (label True) mi trovo ad avere $-y log_2 sigma(z)$ se $y=0$ ho la funzinzone $1-log_2 sigma(z)$.

  Grado delle computazioni, prendo in input la variabile $x$ e $w$ (nel caso del forward la $x$ è variabile e $w$ costante, viceversa nella differenziazione). Esse danno forma alla variabile $z$ (ad esempio prodotto). La catenza forma un grafo ad alto livello. L'ultima parte è quella di decisione, il modello esce con dei valori

  //agiungere immagini
  A partre da $x$ e $w$ viene generata $u = x times w$. A questo punto lo sommo a $b$ (realizzo il livello lineare):
  $
    z = u + b
  $
  (Composizione di funzioni). In uscita avrò $hat(y) = theta(z)$ e infine calcolo la loss.

  Una volta ottenuto il valore di loos metto in atto la discesa del gradiente con la chain rule. La chain rule è data dalla derivata di tutte le variabili in gioco:

  - $t = (alpha l) / (alpha hat(y))$
  - $(alpha l) / a(alpha z)$
  - $(alpha z) / (alpha u)$
  - $(alpha u) / (alpha w)$

  la deriva di $(alpha L)/(alpha w)$ è data dal prodotto di tutte le derivate.
]

== SDG in PyTorch

`loss.backward()` fa tutto lui

`optimezer.step()` ci aggiunge una regola di aggiornamento.

```py
for epoch in range (n_epochs)
  for data, targat in dataloader:
    output = model(data)#costruisce un grafo
    loss = criteration(output, target)
    optimezer.zero_grad() #pulisce i buffer
    loss.backward()
    optimizer.step()#aggiorna i pesi
```

//aggoungere confronto mini-batch e singolo elemento
La disce del graideitne su un singolo elemnto alla volta è molto più smooth, tuttavia è molto costosa a livello computazionale.





