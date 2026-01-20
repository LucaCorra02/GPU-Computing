#import "../template.typ": *

= Lezione 3

// Da integrare con lezione 2
== Lezione 2 proseguo
Non è corretto calcolare i thread e i blocchi con la divisione intera. è meglio fissare il numero di thread e blocchi in modo fisso. Una volta fissato il numero di thread per blocco divido la struttura dati orginale in blocchi usando una divisione e poi faccio una ``` ceil```.

Funzioni ``` device```. Sono funzioni utiizzate internamente al kernel, lavorano esclusivamente sulla GPU. Devono essere marcate con il decoratore ```py @cuda.jit(device=True)```. Inoltre possono avere tipo di ritorno a differenza del kernel. 

#attenzione()[
  Le funzioni device non son un kernel
]

Impostando l'opzione ``` inline = True``` il codice della funzione device viene inserito dal compilatore dentro il kernel, risparmiando il costo della chiamata. 

Di default il trasferimento avviene in maniera automatica, (potremo creare una struttura numpy e passarla indipendentemente a host e GPU). Tuttavia è sempre meglio essere conservativi. Risolve problemi:
- trasferimenti pesanti
- consuma banda
- aumenta la latenza (esecuzione che aspetta i trasferimenti)



