---
commit: aed2ff42
---
Author (2026-06-04): the fast interpreter was not being killed — it had
reached a state that was no longer exciting: adding more n-grams and slowly
improving performance. I got what I wanted from that implementation — rough
data from the results. I started the SSA design because it was time to try
something different, and maybe reach a higher ceiling.
