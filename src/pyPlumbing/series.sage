from sage.all_cmdline import *   # import sage library

import numpy as np
from utils import *
    
class Series():
    def __init__(self, numerical, variables=None):

        # Ensure numerical is properly constructed. It should be a list of monomials.
        # Each monomial should contain a list of powers and a coefficient.

        for monomial in numerical:
            error_message = "Series incorrectly constructed. Please check input."
            example = "Example: [[[1,2,3], 4], [0,3,4], -3] represents 4*z0^1*z1^2*z2^3 -3*z1^3*z2^4"
            assert type(monomial) in [list,tuple], error_message + f" Each monomial should be a list or a tuple." + example
            assert len(monomial) == 2, error_message + f" Each monomial should contain a list of powers and a coefficient." + example
            assert len(monomial[0]) == len(numerical[0][0]), error_message + " All monomials should have the same number of powers." + example
            assert type(monomial[0]) in [list,tuple], error_message + " The first element of each monomial should be a list or a tuple of powers." + example
            assert (not monomial in [Integer, Rational]) or monomial[1].is_constant(), error_message + " The second element of each monomial should be a number. " + example
            assert all([type(power) in [int,Rational] or type for power in monomial[0]]), error_message + " The powers of each monomial should be integers." + example
        
        self._n_variables = len(numerical[0][0]) if len(numerical) > 0 else 0
        self._numerical = sorted([[tuple(p),c] for p,c in numerical if c != 0], key=lambda x: sum(x[0]))
        self._variables = variables if self._n_variables > 0 else []
        self._dictionary = None
        self._symbolic = None
        self._powers = None
        self._coefficients = None



    @classmethod
    def from_symbolic(cls, series):
        """
            Create a Series object from a symbolic expression.
        """
        
        try:
            variables = series.variables()
            series = series.expand()
        except:
            variables = None
    
        numerical = multi_poly_coeffs(series)
        return cls(numerical, variables)
    
    @classmethod
    def from_dictionary(cls, dictionary,variables=None):
        """
            Create a Series object from a dictionary.
        """
        numerical = list(dictionary.items())
        return cls(numerical, variables)

    @classmethod
    def from_function(cls, function, dimension, range, variables=None):
        """
            Create a Series object from a function.
        """
        raise NotImplementedError
        return cls(series, variables)

    @property
    def variables(self):
        """
            Return the variables of the series.
        """
        if self._variables == None:
            self._variables = [var(f"z_{i}") for i in range(self.n_variables)]
        return self._variables

    @property
    def n_variables(self):
        """
            Return the number of variables in the series.
        """
        return self._n_variables
    
    @property
    def max_degree(self, key=None):
        # Return the maximum order of the series for series with one variable
        # If more variables and key=None throw error
        if self.n_variables > 1 and key == None:
            raise ValueError("Please specify a key for the maximum order.")
        if key == None and self.n_variables == 1:
            key = max

        max_order = max([key(monomial[0]) for monomial in self.numerical])    
        return max_order
    
    @property
    def numerical(self):
        """
        Return the numerical representation of the series.
        """
        return self._numerical

    @property
    def dictionary(self):
        """
        Return the dictionary representation of the series.
        """
        if self._dictionary == None:
            self._dictionary = dict()
            for powers, coeff in self.numerical:
                self._dictionary[tuple(powers)] = coeff
        return self._dictionary
    
    @property
    def symbolic(self):
        """
        Return the symbolic representation of the series.
        """
        if self._symbolic == None:
            self._symbolic = 0
            for powers,coeff in self.numerical:
                self._symbolic += coeff * product([v**p for v,p in zip(self.variables,powers)])
        return self._symbolic
    
    @property
    def n_terms(self):
        return len(self.numerical)
    
    @property
    def powers(self):
        if self._powers == None:
            self._powers = [monomial[0] for monomial in self.numerical]
        return self._powers
    
    @property
    def coefficients(self):
        if self._coefficients == None:
            self._coefficients = [monomial[1] for monomial in self.numerical]
        return self._coefficients

    @property
    def min_degree(self, key=None):
        # Return the minimum order of the series for series with one variable
        # If more variables and key=None throw error
        if self.n_variables > 1 and key == None:
            raise ValueError("Please specify a key for the minimum order.")
        if len(self.numerical) == 0:
            return 0
        if key == None and self.n_variables == 1:
            key = min
        min_order = min([key(monomial[0]) for monomial in self.numerical])    
        return min_order

    def collect_like_terms(self):
        """
        Collect like terms in the series.
        """
        result_dict = {}
        for powers, coeff in self.numerical:
            if powers in result_dict:
                powers = round(powers*1.,10)
                result_dict[powers] += coeff
            else:
                result_dict[powers] = coeff
        return Series.from_dictionary(result_dict,self.variables)
    
    def _make_none(self):
        """
        Make all properties None.
        """
        self._numerical = None
        self._dictionary = None
        self._symbolic = None
        self._n_variables = None
        self._powers = None
    
    def _multiply_series(self, other):
        """
        Multiply two multivariate polynomials using NumPy operations.
        Each polynomial is represented as a list of (exponents, coefficient) tuples,
        where exponents are arrays representing the powers of each variable.
        """
        exps1, coeffs1 = zip(*self.numerical)
        exps2, coeffs2 = zip(*other.numerical)
        
        exps1 = np.array(exps1)
        coeffs1 = np.array(coeffs1)
        exps2 = np.array(exps2)
        coeffs2 = np.array(coeffs2)
        
        # Broadcasting addition of exponents
        new_exps = exps1[:, np.newaxis, :] + exps2[np.newaxis, :, :]
        new_coeffs = coeffs1[:, np.newaxis] * coeffs2[np.newaxis, :]
        
        # Reshape the arrays to be a list of tuples
        new_exps = new_exps.reshape(-1, exps1.shape[1])
        new_coeffs = new_coeffs.flatten()
        
        # Sum the coefficients of the same exponents
        result_dict = {}
        for exp, coeff in zip(map(tuple, new_exps), new_coeffs):
            if exp in result_dict:
                result_dict[exp] += coeff
            else:
                result_dict[exp] = coeff
        
        # Convert the result to the required format, filtering out zero coefficients
        result = [(list(exp), coeff) for exp, coeff in result_dict.items() if coeff != 0]
        return Series(result,self.variables)
    

    def _fft_multiply_series(self, other):
        """
        Multiply two multivariate polynomials using FFT.
        Each polynomial is represented as a list of (exponents, coefficient) tuples,
        where exponents are arrays representing the powers of each variable.
        The shape parameter defines the size of the FFT grid.
        """
        # Define the shape of the FFT grid (based on the maximum exponents +1)
        shape = tuple(max(max(e) for e in zip(*self.powers, *other.powers)) + 1 for _ in range(self.n_variables))
        
        # Create coefficient grids
        coeff_grid1 = np.zeros(shape, dtype=complex)
        coeff_grid2 = np.zeros(shape, dtype=complex)
        
        for exps, coeff in self.numerical:
            coeff_grid1[tuple(exps)] = coeff
        
        for exps, coeff in other.numerical:
            coeff_grid2[tuple(exps)] = coeff
        
        # Perform FFT on both grids
        fft_grid1 = np.fft.fftn(coeff_grid1)
        fft_grid2 = np.fft.fftn(coeff_grid2)
        
        # Point-wise multiply the FFT results
        fft_result = fft_grid1 * fft_grid2
        
        # Perform inverse FFT to get the result polynomial coefficients
        result_grid = np.fft.ifftn(fft_result).real
        
        # Extract non-zero coefficients and their exponents
        result = [(list(index), coeff) for index, coeff in np.ndenumerate(result_grid) if coeff != 0]
        
        return result

    
    def __add__(self, other):
        if self.n_variables != other.n_variables:
            raise ValueError("The number of variables of the two series do not match.")
        
        # Combine the two series
        new_dict = join_dicts_with_function(self.dictionary, other.dictionary, lambda x,y: x+y)
        new_numerical = list(new_dict.items())
        return Series(new_numerical, self.variables)
    
    def __sub__(self, other):
        if self.n_variables != other.n_variables and 0 not in [self.n_variables, other.n_variables]:
            raise ValueError("The number of variables of the two series do not match.")
        
        # Combine the two series
        new_dict = join_dicts_with_function(self.dictionary, other.dictionary, lambda x,y: x-y)
        new_numerical = [(k,v) for k,v in new_dict.items() if v != 0]
        return Series(new_numerical, self.variables)
    
    def __mul__(self, other):
        if self.n_variables != other.n_variables:
            return Series.from_symbolic(self.symbolic * other.symbolic)
            raise ValueError("The number of variables of the two series do not match.")
        
        # Decide which multiplication method to use based on the number of terms
        if True or self.n_terms * other.n_terms < 1000:
            #print("Using NumPy multiplication")
            res = Series.from_symbolic(self.symbolic * other.symbolic)
        else:
            # For some reason fft is super slow
            print("Using FFT multiplication")
            res = self._fft_multiply_series(other)
        return res

    def __repr__(self):
        # Check if all powers are equal mod 1
        return self.show(max_terms = 10)
    
    def show(self,max_terms=10):
        # Check if all powers are equal mod 1
        res = str("")
        if self.variables == (var("q"),):
            rational_part = lambda x: x - floor(x)
            if all(rational_part(self.powers[0][0]) == rational_part(x[0]) for x in self.powers):
                overall = self.powers[0][0]
                if overall != 0:
                    res += f"q^({overall})(" 
                n_added = 0
                for powers, coeff in self.numerical:
                    if coeff == 0:
                        continue
                    if n_added >= max_terms:
                        break
                    if coeff < 0 and powers != self.powers[0]:
                        res = res[:-3] + f" - "
                    if powers[0] == overall:
                        res +=str(coeff) + " + "
                    else:
                        res += f"{abs(coeff)}q^({powers[0]-overall}) + "
                    n_added += 1
                try:
                    res += f"O(q^{self.numerical[max_terms][0][0]-overall})"
                except:
                    res += f"O(q^{self.numerical[n_added-1][0][0]+1-overall})"
                if overall != 0:
                    res += f")"
                return res
        return str(self.symbolic)
    
    def pow(self,power):
        assert type(power) in [int, np.int64, Integer], "The power should be a positive integer."
        assert power >= 0, "The power should be a positive integer."
        
        if power == 0:
            return Series.from_dictionary({tuple([0]*self.n_variables):1},self.variables)
        if power == 1:
            return self
        return Series.from_symbolic(self.symbolic**power)
    
    def truncate(self, max_order):
        """
        Truncate the series to a maximum order.
        """
        new_numerical = [monomial for monomial in self.numerical if sum(monomial[0]) < max_order]
        return Series(new_numerical, self.variables)