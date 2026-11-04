<!-- command=lta eval regress --scorers lm,finbert,llm --se both --window post_h55 | run_id=0398465b-b81d-4324-9536-74af26d46fa3 | git_sha=68f21d2eb28ab73b547aabe6f34d3370b6c0ff0b-dirty | provider=licensed -->

| scorer | window | horizon | spec | se | coef_score_z | se_score_z | t_score_z | nobs | periods | r2 |
|---|---|---|---|---|---|---|---|---|---|---|
| lm | post_h55 | CAR[0,1] | score_only | twoway | 0.000337664 | 0.00173932 | 0.194136 | 937 | 39 | 3.83017e-05 |
| lm | post_h55 | CAR[0,1] | score_only | fm | 0.000599022 | 0.000838267 | 0.714595 | 937 | 2 |  |
| lm | post_h55 | CAR[0,1] | controls_no_sue | twoway | 0.000265262 | 0.00170087 | 0.155957 | 920 | 39 | 0.000246211 |
| lm | post_h55 | CAR[0,1] | controls_no_sue | fm | 0.000491504 | 0.000668707 | 0.735006 | 920 | 2 |  |
| lm | post_h55 | CAR[0,1] | controls | twoway | 0.00036065 | 0.00181334 | 0.198887 | 887 | 39 | 0.00198326 |
| lm | post_h55 | CAR[0,1] | controls | fm | 0.00056929 | 0.000869787 | 0.654517 | 887 | 2 |  |
| lm | post_h55 | CAR[0,1] | controls_bm | twoway | 0.000812709 | 0.00179594 | 0.452526 | 887 | 39 | 0.0084134 |
| lm | post_h55 | CAR[0,1] | controls_bm | fm | 0.000977995 | 0.000845609 | 1.15656 | 887 | 2 |  |
| lm | post_h55 | CAR[0,1] | controls_sue_announced | twoway | 0.000134164 | 0.00190372 | 0.0704744 | 850 | 33 | 0.000939141 |
| lm | post_h55 | CAR[0,1] | controls_sue_announced | fm | 0.000603292 | 0.00111315 | 0.54197 | 850 | 2 |  |
| lm | post_h55 | CAR[0,1] | sue_orth | twoway | 0.000184777 | 0.00180459 | 0.102393 | 887 | 39 | 0.000414222 |
| lm | post_h55 | CAR[0,1] | sue_orth | fm | 0.000552859 | 0.000835768 | 0.661499 | 887 | 2 |  |
| lm | post_h55 | CAR[0,5] | score_only | twoway | -0.0011026 | 0.00240887 | -0.457724 | 937 | 39 | 0.000226578 |
| lm | post_h55 | CAR[0,5] | score_only | fm | -0.000568795 | 0.00173135 | -0.328527 | 937 | 2 |  |
| lm | post_h55 | CAR[0,5] | controls_no_sue | twoway | -0.0010938 | 0.00232619 | -0.470211 | 920 | 39 | 0.015368 |
| lm | post_h55 | CAR[0,5] | controls_no_sue | fm | -0.000390097 | 0.00141154 | -0.276362 | 920 | 2 |  |
| lm | post_h55 | CAR[0,5] | controls | twoway | -0.000935129 | 0.00255644 | -0.365793 | 887 | 39 | 0.00594305 |
| lm | post_h55 | CAR[0,5] | controls | fm | -2.63818e-05 | 0.00162161 | -0.0162689 | 887 | 2 |  |
| lm | post_h55 | CAR[0,5] | controls_bm | twoway | -0.000522517 | 0.0025667 | -0.203576 | 887 | 39 | 0.00897856 |
| lm | post_h55 | CAR[0,5] | controls_bm | fm | 0.000417595 | 0.00170385 | 0.245089 | 887 | 2 |  |
| lm | post_h55 | CAR[0,5] | controls_sue_announced | twoway | -0.00115266 | 0.00272479 | -0.423025 | 850 | 33 | 0.00395216 |
| lm | post_h55 | CAR[0,5] | controls_sue_announced | fm | -5.01072e-05 | 0.0015609 | -0.0321016 | 850 | 2 |  |
| lm | post_h55 | CAR[0,5] | sue_orth | twoway | -0.00108972 | 0.00250309 | -0.43535 | 887 | 39 | 0.00359234 |
| lm | post_h55 | CAR[0,5] | sue_orth | fm | -7.62786e-05 | 0.0015951 | -0.0478206 | 887 | 2 |  |
| lm | post_h55 | CAR[0,20] | score_only | twoway | 0.00141978 | 0.00325177 | 0.436619 | 937 | 39 | 0.000161335 |
| lm | post_h55 | CAR[0,20] | score_only | fm | 0.00161473 | 0.000599639 | 2.69284 | 937 | 2 |  |
| lm | post_h55 | CAR[0,20] | controls_no_sue | twoway | 0.00130412 | 0.00311792 | 0.418264 | 920 | 39 | 0.00656685 |
| lm | post_h55 | CAR[0,20] | controls_no_sue | fm | 0.00155879 | 0.000218583 | 7.13134 | 920 | 2 |  |
| lm | post_h55 | CAR[0,20] | controls | twoway | 0.00248297 | 0.00317905 | 0.781043 | 887 | 39 | 0.0109142 |
| lm | post_h55 | CAR[0,20] | controls | fm | 0.00322613 | 0.000474897 | 6.79333 | 887 | 2 |  |
| lm | post_h55 | CAR[0,20] | controls_bm | twoway | 0.00384515 | 0.00324094 | 1.18643 | 887 | 39 | 0.0249909 |
| lm | post_h55 | CAR[0,20] | controls_bm | fm | 0.00458612 | 0.000668099 | 6.86444 | 887 | 2 |  |
| lm | post_h55 | CAR[0,20] | controls_sue_announced | twoway | 0.00372635 | 0.00318304 | 1.17069 | 850 | 33 | 0.0096479 |
| lm | post_h55 | CAR[0,20] | controls_sue_announced | fm | 0.00477185 | 0.000499671 | 9.54999 | 850 | 2 |  |
| lm | post_h55 | CAR[0,20] | sue_orth | twoway | 0.00236422 | 0.00323791 | 0.730169 | 887 | 39 | 0.00104642 |
| lm | post_h55 | CAR[0,20] | sue_orth | fm | 0.00305834 | 0.000455238 | 6.71812 | 887 | 2 |  |
| lm | post_h55 | CAR[0,60] | score_only | twoway | -0.00366798 | 0.00427836 | -0.857335 | 937 | 39 | 0.00061201 |
| lm | post_h55 | CAR[0,60] | score_only | fm | -0.00356339 | 0.000310397 | -11.4801 | 937 | 2 |  |
| lm | post_h55 | CAR[0,60] | controls_no_sue | twoway | -0.00326045 | 0.00410852 | -0.793583 | 920 | 39 | 0.0137438 |
| lm | post_h55 | CAR[0,60] | controls_no_sue | fm | -0.00260456 | 0.000442943 | -5.88011 | 920 | 2 |  |
| lm | post_h55 | CAR[0,60] | controls | twoway | -0.00159996 | 0.00455329 | -0.351386 | 887 | 39 | 0.0110529 |
| lm | post_h55 | CAR[0,60] | controls | fm | -0.000213348 | 0.000363934 | -0.586229 | 887 | 2 |  |
| lm | post_h55 | CAR[0,60] | controls_bm | twoway | -0.000106442 | 0.00463695 | -0.0229552 | 887 | 39 | 0.0206158 |
| lm | post_h55 | CAR[0,60] | controls_bm | fm | 0.00113432 | 0.000452479 | 2.5069 | 887 | 2 |  |
| lm | post_h55 | CAR[0,60] | controls_sue_announced | twoway | -0.000611174 | 0.00443582 | -0.137781 | 850 | 33 | 0.00583942 |
| lm | post_h55 | CAR[0,60] | controls_sue_announced | fm | 0.0010601 | 0.000550493 | 1.92573 | 850 | 2 |  |
| lm | post_h55 | CAR[0,60] | sue_orth | twoway | -0.00154821 | 0.00463045 | -0.334354 | 887 | 39 | 0.00521399 |
| lm | post_h55 | CAR[0,60] | sue_orth | fm | -0.000419115 | 0.000348787 | -1.20164 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,1] | score_only | twoway | 0.00134639 | 0.00158718 | 0.848289 | 937 | 39 | 0.000616487 |
| finbert | post_h55 | CAR[0,1] | score_only | fm | 0.0014109 | 0.00054324 | 2.5972 | 937 | 2 |  |
| finbert | post_h55 | CAR[0,1] | controls_no_sue | twoway | 0.00123095 | 0.00150635 | 0.817174 | 920 | 39 | 0.000758132 |
| finbert | post_h55 | CAR[0,1] | controls_no_sue | fm | 0.00132078 | 0.000450452 | 2.93213 | 920 | 2 |  |
| finbert | post_h55 | CAR[0,1] | controls | twoway | 0.00163483 | 0.00155903 | 1.04862 | 887 | 39 | 0.00289311 |
| finbert | post_h55 | CAR[0,1] | controls | fm | 0.00148056 | 0.000493305 | 3.00132 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,1] | controls_bm | twoway | 0.00206613 | 0.00162733 | 1.26965 | 887 | 39 | 0.0096912 |
| finbert | post_h55 | CAR[0,1] | controls_bm | fm | 0.00185514 | 0.000577867 | 3.21033 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,1] | controls_sue_announced | twoway | 0.00188583 | 0.00164375 | 1.14727 | 850 | 33 | 0.00216973 |
| finbert | post_h55 | CAR[0,1] | controls_sue_announced | fm | 0.00196451 | 0.000386185 | 5.08698 | 850 | 2 |  |
| finbert | post_h55 | CAR[0,1] | sue_orth | twoway | 0.00151464 | 0.00153352 | 0.987688 | 887 | 39 | 0.0012172 |
| finbert | post_h55 | CAR[0,1] | sue_orth | fm | 0.00149305 | 0.0004485 | 3.32898 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,5] | score_only | twoway | -0.00216895 | 0.00197182 | -1.09998 | 937 | 39 | 0.000887601 |
| finbert | post_h55 | CAR[0,5] | score_only | fm | -0.00205477 | 0.000343216 | -5.98682 | 937 | 2 |  |
| finbert | post_h55 | CAR[0,5] | controls_no_sue | twoway | -0.00263032 | 0.00152171 | -1.72853 | 920 | 39 | 0.0164867 |
| finbert | post_h55 | CAR[0,5] | controls_no_sue | fm | -0.00247802 | 0.000216857 | -11.427 | 920 | 2 |  |
| finbert | post_h55 | CAR[0,5] | controls | twoway | -0.00205793 | 0.00159561 | -1.28974 | 887 | 39 | 0.00662339 |
| finbert | post_h55 | CAR[0,5] | controls | fm | -0.00192827 | 0.000144761 | -13.3204 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,5] | controls_bm | twoway | -0.0016856 | 0.00157569 | -1.06975 | 887 | 39 | 0.00949415 |
| finbert | post_h55 | CAR[0,5] | controls_bm | fm | -0.00159971 | 0.000149415 | -10.7065 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,5] | controls_sue_announced | twoway | -0.00226457 | 0.00153211 | -1.47808 | 850 | 33 | 0.00471762 |
| finbert | post_h55 | CAR[0,5] | controls_sue_announced | fm | -0.00174588 | 4.12213e-05 | -42.3537 | 850 | 2 |  |
| finbert | post_h55 | CAR[0,5] | sue_orth | twoway | -0.00213405 | 0.00156828 | -1.36076 | 887 | 39 | 0.00426725 |
| finbert | post_h55 | CAR[0,5] | sue_orth | fm | -0.00186648 | 0.000176293 | -10.5874 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,20] | score_only | twoway | -0.00481769 | 0.0033081 | -1.45633 | 937 | 39 | 0.0018806 |
| finbert | post_h55 | CAR[0,20] | score_only | fm | -0.00431776 | 0.000180226 | -23.9574 | 937 | 2 |  |
| finbert | post_h55 | CAR[0,20] | controls_no_sue | twoway | -0.00593628 | 0.00356113 | -1.66697 | 920 | 39 | 0.00931844 |
| finbert | post_h55 | CAR[0,20] | controls_no_sue | fm | -0.00534319 | 1.77533e-05 | -300.968 | 920 | 2 |  |
| finbert | post_h55 | CAR[0,20] | controls | twoway | -0.00515758 | 0.00357819 | -1.44139 | 887 | 39 | 0.0126739 |
| finbert | post_h55 | CAR[0,20] | controls | fm | -0.00449758 | 5.67522e-06 | -792.495 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,20] | controls_bm | twoway | -0.00397157 | 0.00350246 | -1.13393 | 887 | 39 | 0.0250678 |
| finbert | post_h55 | CAR[0,20] | controls_bm | fm | -0.00348298 | 3.53153e-06 | -986.252 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,20] | controls_sue_announced | twoway | -0.00426569 | 0.00364814 | -1.16928 | 850 | 33 | 0.0101123 |
| finbert | post_h55 | CAR[0,20] | controls_sue_announced | fm | -0.00320646 | 1.34595e-05 | -238.231 | 850 | 2 |  |
| finbert | post_h55 | CAR[0,20] | sue_orth | twoway | -0.00465887 | 0.00353316 | -1.31861 | 887 | 39 | 0.00242111 |
| finbert | post_h55 | CAR[0,20] | sue_orth | fm | -0.00427604 | 9.53246e-06 | -448.577 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,60] | score_only | twoway | -0.00944498 | 0.00340822 | -2.77124 | 937 | 39 | 0.00410808 |
| finbert | post_h55 | CAR[0,60] | score_only | fm | -0.00910176 | 0.000259292 | -35.1023 | 937 | 2 |  |
| finbert | post_h55 | CAR[0,60] | controls_no_sue | twoway | -0.0109649 | 0.00320365 | -3.42263 | 920 | 39 | 0.018899 |
| finbert | post_h55 | CAR[0,60] | controls_no_sue | fm | -0.0105454 | 0.000845894 | -12.4666 | 920 | 2 |  |
| finbert | post_h55 | CAR[0,60] | controls | twoway | -0.0106644 | 0.00358565 | -2.9742 | 887 | 39 | 0.0164743 |
| finbert | post_h55 | CAR[0,60] | controls | fm | -0.00972888 | 0.000539372 | -18.0374 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,60] | controls_bm | twoway | -0.0093672 | 0.00356434 | -2.62803 | 887 | 39 | 0.0248534 |
| finbert | post_h55 | CAR[0,60] | controls_bm | fm | -0.00859628 | 0.000770913 | -11.1508 | 887 | 2 |  |
| finbert | post_h55 | CAR[0,60] | controls_sue_announced | twoway | -0.00989208 | 0.00342159 | -2.89107 | 850 | 33 | 0.0105379 |
| finbert | post_h55 | CAR[0,60] | controls_sue_announced | fm | -0.00861061 | 0.000764499 | -11.2631 | 850 | 2 |  |
| finbert | post_h55 | CAR[0,60] | sue_orth | twoway | -0.00994423 | 0.00350493 | -2.83722 | 887 | 39 | 0.00988453 |
| finbert | post_h55 | CAR[0,60] | sue_orth | fm | -0.00944421 | 0.000573734 | -16.461 | 887 | 2 |  |
| llm | post_h55 | CAR[0,1] | score_only | twoway | 0.00110743 | 0.00219002 | 0.505673 | 937 | 39 | 0.000391451 |
| llm | post_h55 | CAR[0,1] | score_only | fm | 0.000500294 | 0.00210018 | 0.238215 | 937 | 2 |  |
| llm | post_h55 | CAR[0,1] | controls_no_sue | twoway | 0.00100096 | 0.00209988 | 0.476676 | 920 | 39 | 0.000535529 |
| llm | post_h55 | CAR[0,1] | controls_no_sue | fm | 0.000406054 | 0.00201724 | 0.201292 | 920 | 2 |  |
| llm | post_h55 | CAR[0,1] | controls | twoway | 0.00225166 | 0.00212487 | 1.05967 | 887 | 39 | 0.00344229 |
| llm | post_h55 | CAR[0,1] | controls | fm | 0.00105716 | 0.00182183 | 0.580271 | 887 | 2 |  |
| llm | post_h55 | CAR[0,1] | controls_bm | twoway | 0.0030247 | 0.00228357 | 1.32455 | 887 | 39 | 0.0108316 |
| llm | post_h55 | CAR[0,1] | controls_bm | fm | 0.00166786 | 0.00188589 | 0.884388 | 887 | 2 |  |
| llm | post_h55 | CAR[0,1] | controls_sue_announced | twoway | 0.00200244 | 0.00218709 | 0.915572 | 850 | 33 | 0.00196153 |
| llm | post_h55 | CAR[0,1] | controls_sue_announced | fm | 0.00131796 | 0.00246543 | 0.534577 | 850 | 2 |  |
| llm | post_h55 | CAR[0,1] | sue_orth | twoway | 0.00179158 | 0.00219114 | 0.817648 | 887 | 39 | 0.00134393 |
| llm | post_h55 | CAR[0,1] | sue_orth | fm | 0.00103714 | 0.00175574 | 0.590713 | 887 | 2 |  |
| llm | post_h55 | CAR[0,5] | score_only | twoway | 0.000119286 | 0.00232761 | 0.0512482 | 937 | 39 | 2.51973e-06 |
| llm | post_h55 | CAR[0,5] | score_only | fm | -0.000386931 | 0.00162966 | -0.23743 | 937 | 2 |  |
| llm | post_h55 | CAR[0,5] | controls_no_sue | twoway | -0.00104516 | 0.00238638 | -0.437968 | 920 | 39 | 0.015323 |
| llm | post_h55 | CAR[0,5] | controls_no_sue | fm | -0.00225803 | 0.00107047 | -2.10939 | 920 | 2 |  |
| llm | post_h55 | CAR[0,5] | controls | twoway | 0.000458425 | 0.00250761 | 0.182813 | 887 | 39 | 0.00579967 |
| llm | post_h55 | CAR[0,5] | controls | fm | -0.00100652 | 0.00101716 | -0.989538 | 887 | 2 |  |
| llm | post_h55 | CAR[0,5] | controls_bm | twoway | 0.00114892 | 0.00241414 | 0.475912 | 887 | 39 | 0.00914035 |
| llm | post_h55 | CAR[0,5] | controls_bm | fm | -0.000336395 | 0.000910507 | -0.369459 | 887 | 2 |  |
| llm | post_h55 | CAR[0,5] | controls_sue_announced | twoway | -0.000943587 | 0.00226095 | -0.417342 | 850 | 33 | 0.00384104 |
| llm | post_h55 | CAR[0,5] | controls_sue_announced | fm | -0.00187108 | 0.00158822 | -1.1781 | 850 | 2 |  |
| llm | post_h55 | CAR[0,5] | sue_orth | twoway | 0.000142915 | 0.00250155 | 0.0571305 | 887 | 39 | 0.00335356 |
| llm | post_h55 | CAR[0,5] | sue_orth | fm | -0.000942104 | 0.000981085 | -0.960268 | 887 | 2 |  |
| llm | post_h55 | CAR[0,20] | score_only | twoway | -0.00835849 | 0.0044557 | -1.87591 | 937 | 39 | 0.00531293 |
| llm | post_h55 | CAR[0,20] | score_only | fm | -0.0079559 | 0.00049264 | -16.1495 | 937 | 2 |  |
| llm | post_h55 | CAR[0,20] | controls_no_sue | twoway | -0.0112638 | 0.00462842 | -2.43362 | 920 | 39 | 0.0156423 |
| llm | post_h55 | CAR[0,20] | controls_no_sue | fm | -0.0114216 | 0.000890145 | -12.8312 | 920 | 2 |  |
| llm | post_h55 | CAR[0,20] | controls | twoway | -0.00691331 | 0.00458908 | -1.50647 | 887 | 39 | 0.0138008 |
| llm | post_h55 | CAR[0,20] | controls | fm | -0.00701559 | 0.00214556 | -3.26982 | 887 | 2 |  |
| llm | post_h55 | CAR[0,20] | controls_bm | twoway | -0.00493867 | 0.00451714 | -1.09332 | 887 | 39 | 0.0254253 |
| llm | post_h55 | CAR[0,20] | controls_bm | fm | -0.00496862 | 0.00257177 | -1.93199 | 887 | 2 |  |
| llm | post_h55 | CAR[0,20] | controls_sue_announced | twoway | -0.00763857 | 0.00518181 | -1.47411 | 850 | 33 | 0.0122454 |
| llm | post_h55 | CAR[0,20] | controls_sue_announced | fm | -0.00740986 | 0.00120108 | -6.16932 | 850 | 2 |  |
| llm | post_h55 | CAR[0,20] | sue_orth | twoway | -0.00668127 | 0.00442945 | -1.50837 | 887 | 39 | 0.00371999 |
| llm | post_h55 | CAR[0,20] | sue_orth | fm | -0.00668901 | 0.00211954 | -3.15588 | 887 | 2 |  |
| llm | post_h55 | CAR[0,60] | score_only | twoway | -0.000859515 | 0.0063764 | -0.134796 | 937 | 39 | 3.19303e-05 |
| llm | post_h55 | CAR[0,60] | score_only | fm | 0.000117739 | 0.00232341 | 0.050675 | 937 | 2 |  |
| llm | post_h55 | CAR[0,60] | controls_no_sue | twoway | -0.00477638 | 0.00673333 | -0.709363 | 920 | 39 | 0.0141934 |
| llm | post_h55 | CAR[0,60] | controls_no_sue | fm | -0.00584886 | 0.00404245 | -1.44686 | 920 | 2 |  |
| llm | post_h55 | CAR[0,60] | controls | twoway | -0.00183943 | 0.00712953 | -0.258002 | 887 | 39 | 0.011064 |
| llm | post_h55 | CAR[0,60] | controls | fm | -0.00140588 | 0.00508241 | -0.276617 | 887 | 2 |  |
| llm | post_h55 | CAR[0,60] | controls_bm | twoway | 0.000543035 | 0.0070121 | 0.0774426 | 887 | 39 | 0.0206269 |
| llm | post_h55 | CAR[0,60] | controls_bm | fm | 0.00081692 | 0.00519312 | 0.157308 | 887 | 2 |  |
| llm | post_h55 | CAR[0,60] | controls_sue_announced | twoway | -0.00325193 | 0.00843773 | -0.385403 | 850 | 33 | 0.00619868 |
| llm | post_h55 | CAR[0,60] | controls_sue_announced | fm | -0.00341177 | 0.00460055 | -0.741601 | 850 | 2 |  |
| llm | post_h55 | CAR[0,60] | sue_orth | twoway | -0.000891 | 0.00675134 | -0.131974 | 887 | 39 | 0.00512819 |
| llm | post_h55 | CAR[0,60] | sue_orth | fm | -0.000950961 | 0.00497053 | -0.19132 | 887 | 2 |  |
