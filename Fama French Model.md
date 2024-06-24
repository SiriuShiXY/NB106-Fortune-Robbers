There are 3 factor model and 5 factor model. The latter first appeared in their publication in 2014, which is developed based on the former, released in the 1990s.

To understand Fama French(FF), you have to know CAPM(Capital Asset Pricing Model).

## CAPM
### Assumptions
1. The market is in a competitive equilibrium; 
2. Common single-period investment horizon;
3. All assets are tradable (market portfolio); 
4. No transaction costs, no taxes; 
5. Lending and borrowing at a common risk-free rate are unlimited.
6. Investors are rational mean-variance optimizers 
7. Homogeneous expectations
### The Market Portfolio
$$
p_{i}=\text{price of one share of risky asset }i

$$$$
n_{i}=\text{number of shares outstanding for risky asset }i

$$$$
\omega_{im}={\frac{{p_{i}*n_{i}}}{\sum_{i}p_{i}n_{i}}}={\frac{{\text{Market Capitalization of Security } }i}{\text{Total Market Capitalization}}}
$$
In other words, market portfolio is the portfolio that contains corresponding proportions of each security.

Investors are expected to hold different combinations of the market portfolio and the riskless asset.

Capital Market Line
$$
E(R_{p})=R_{f}+\left( \frac{{E(R_{M})-R_{f}}}{\sigma_{M} }\right)\sigma_{p}
$$
![[Pasted image 20240624141519.png]]
![[Pasted image 20240624142008.png]]
![[Pasted image 20240624142030.png]]
With this in your mind, you can calculate beta

Now, let's go back to FF.

## Three Factors
### They are
1. Market excess return
2. Outperformance of small versus big companies
3. Outperformance of high book/market versus low book/market companies

$$
r=R_{f}+\beta(R_{m}-R_{f})+b_{s}·SMB+b_{v}·HML+\alpha

$$
r is the portfolio's expected rate of return, Rf is the risk-free return rate, and Rm is the return of the market portfolio. The "three factor" β is analogous to the classical β but not equal to it, since there are now two additional factors to do some of the work. SMB stands for "Small market capitalization Minus Big" and HML for "High book-to-market ratio Minus Low"; they measure the historic excess returns of small caps over big caps and of value stocks over growth stocks, alpha is the error term.

### notes
FF 3 factor explains over 90%of the diversified porfolios returns, higher than the 70% for CAPM. Generally, this supports their observation that two classes of stockes outperform the market as a whole: (a) small caps and (b) those with high B/M ratio
(values stocks)

local factors vary from states to states

**maybe we should look into that in china?**

## Five Factors

adding profitability and investment

profitability: difference between the returns of firms with robust (high) and weak (low) operating profitability  RMW
investment: conservative minus aggressive  CMA

### notes
HML seems redundant given RMW (0.7 correlation), and it is completely explained by the rest four factors

Could use the below test to help benchmark our portfolio

Gibbons, Michael R., et al. “A Test of the Efficiency of a Given Portfolio.” _Econometrica_, vol. 57, no. 5, 1989, pp. 1121–52. _JSTOR_, https://doi.org/10.2307/1913625. Accessed 24 June 2024.