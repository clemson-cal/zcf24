from dataclasses import dataclass, field
from numpy import *

__all__ = ["DiskModel", "planck_spectrum", "eddington_accretion_rate"]


# Relevant physical constants in CGS units
M_sun = 1.9884e33
sigma_t = 5.670374419e-5
G = 6.6743e-8
c = 2.99792458e10
h = 6.62619650e-27
k = 1.38062259e-16
parsec = 3.086e18
day = 86400
month = 30 * day
year = 365.24 * day
AU = 1.496e13
eV = 1.60218e-12


@dataclass
class DiskModel:
    """
    Generates variable disk structure using a two-zone disk approximation

    This class can generate surface density profiles, accretion rates, and
    emission spectra based on self-similar solutions of a disk surrounding
    a binary undergoing gravitational wave driven orbital decay.
    """

    n: float = 0.0  # power-law index defining the viscosity law
    mdot: float = 1.0  # mass accretion rate
    tdec: float = 1.0  # viscous decoupling time
    rdec: float = 1.0  # viscous decoupling radius
    ell: float = 1.0  # dimensionless torque parameter
    kappa: float = 5 / 3  # fudge factor, fixed empirically
    binary_mass: float = 1.0  # mass of the binary, and of the merger remnant

    @property
    def isco_radius(self):
        return 6 * G * self.binary_mass / c**2

    def binary_separation(self, t: float) -> float:
        """
        Return the binary separation at a time before the merger

        The binary is at separation rdec when t = -tdec, and merges at t = 0.

        t: time (must be negative)
        """
        rdec, tdec = self.rdec, self.tdec
        return rdec * (-t / tdec) ** (1 / 4)

    def r_star(self, t: float) -> float:
        """
        Return the disk inner edge

        For a positively torqued binary, it means the radius where the surface
        density would go through zero; the disk inner edge can be formally
        either larger or smaller than a. For a negatively torqued binary the
        surface density diverges at r=0, so we define the inner edge to be at
        r=a. If t > 0, the disk inner edge is set to the ISCO for a
        non-rotating BH.
        """
        if t >= 0.0:
            return self.isco_radius
        elif self.ell >= 0.0:
            return self.binary_separation(t) * self.ell**2
        else:
            return self.binary_separation(t)

    def viscosity(self, r: float) -> float:
        """
        Return the kinematic viscosity at radius r in the disk.

        r: radial coordinate in the disk
        """
        rdec, tdec, n = self.rdec, self.tdec, self.n
        nu_dec = rdec**2 / tdec / 6
        return nu_dec * (r / rdec) ** n

    def radius_of_influence_pre_merger(self, t: float) -> float:
        """
        Return the "viscous radius" at some time before the merger

        t: time (must be negative)
        """
        rdec, n = self.rdec, self.n
        a = self.binary_separation(t)
        beta = 4 / (2 - n)
        return rdec * (a / rdec) ** beta

    def radius_of_influence_post_merger(self, t: float) -> float:
        """
        Return the "viscous radius" at some time after the merger

        t: time (must be positive)
        """
        rdec, tdec, kappa, n = self.rdec, self.tdec, self.kappa, self.n
        nu_dec = rdec**2 / tdec / 6
        return (6 * kappa * nu_dec * t / rdec**n) ** (1 / (2 - n))

    def radius_of_influence(self, t: float) -> float:
        """
        Return the "viscous radius" at a time before or after the merger

        A time less than zero implies time before merger
        """
        if t < 0.0:
            return self.radius_of_influence_pre_merger(t)
        else:
            return self.radius_of_influence_post_merger(t)

    def accretion_rate_pre_merger(self, t: float) -> float:
        """
        Return the pre-merger binary accretion rate

        t: time (must be negative)
        """
        ell, mdot, tdec, n = self.ell, self.mdot, self.tdec, self.n
        p = (2 + n) / (2 - n)
        return mdot * (1 - ell * (-t / tdec) ** (-p / 8)) ** (1 / p)

    def accretion_rate_post_merger(self, t: float) -> float:
        """
        Return the post-merger binary accretion rate

        t: time (must be positive)
        """
        tdec, mdot, kappa, n, ell = self.tdec, self.mdot, self.kappa, self.n, self.ell
        p = (2 + n) / (2 - n)
        q = (2 + n) / 4
        return mdot * (1 - ell * (kappa * t / tdec) ** (-p / 8)) ** (1 / q)

    def accretion_rate(self, t: float) -> float:
        """
        Return the pre or post-merger (binary or remnant) accretion rate

        A time less than zero implies time before merger
        """
        if t < 0.0:
            return self.accretion_rate_pre_merger(t)
        else:
            return self.accretion_rate_post_merger(t)

    def surface_density_pre_merger_in(self, r: float, t: float) -> float:
        """
        Return the pre-merger disk surface density within the viscous radius

        r: radial coordinate in the disk (must be less than r_nu_pre)
        t: time (must be negative)
        """
        nu = self.viscosity(r)
        a = self.binary_separation(t)
        ell = self.ell
        return (
            self.accretion_rate_pre_merger(t)
            / (3 * pi * nu)
            * (1 - ell * (a / r) ** 0.5)
        )

    def surface_density_post_merger_in(self, r: float, t: float, isco=False) -> float:
        """
        Return the post-merger disk surface density within the viscous radius

        Note: if isco=True, then the inward Jdot associated with finite ISCO
        radius is accuonted for here. However the post-merger accretion rate
        as predicted in Z+24 does not account for the finite ISCO, rather the
        post-merger Mdot is figured out assuming the disk has a zero Jdot
        post-merger. It implies there would be a small kink in the surface
        density at the radius of influence, if isco=True.

        r: radial coordinate in the disk (must be less than r_nu_post)
        t: time (must be positive)
        """
        sigma0 = self.accretion_rate_post_merger(t) / (3 * pi * self.viscosity(r))
        if isco:
            return sigma0 * (1 - (self.r_star(t) / r) ** 0.5)
        else:
            return sigma0

    def surface_density_out(self, r: float) -> float:
        """
        Return the disk surface density beyond the radius of influence

        r: radial coordinate in the disk (must be less than r_nu)
        """
        nu = self.viscosity(r)
        ell, mdot, rdec, n = self.ell, self.mdot, self.rdec, self.n
        q = (2 + n) / 4
        return mdot / (3 * pi * nu) * (1 - ell * (r / rdec) ** (-q / 2)) ** (1 / q)

    def surface_density(
        self,
        r: float,
        t: float,
        torque_effect: bool = True,
        isco: bool = False,
    ) -> float:
        """
        Return the pre or post-merger (binary or remnant) accretion rate
        """
        if not torque_effect:
            nu = self.viscosity(r)
            if t < 0.0:
                a = self.binary_separation(t)
                return self.mdot / (3 * pi * nu) * (1 - self.ell * (a / r) ** 0.5)
            else:
                return self.mdot / (3 * pi * nu) * (1 - (self.r_star(t) / r) ** 0.5)
        elif r < self.radius_of_influence(t):
            if t < 0.0:
                return self.surface_density_pre_merger_in(r, t)
            else:
                return self.surface_density_post_merger_in(r, t, isco=isco)
        else:
            return self.surface_density_out(r)

    def viscous_power_density(
        self,
        r: float,
        t: float,
        torque_effect: bool = True,
        isco: bool = False,
    ) -> float:
        """
        Return the power per unit surface area dissipated by viscous forces.

        This expression can be found in Eqn. 3.10 of Pringle (1981), and is
        equivalent to 9/8 nu Sigma Omega^2.

        Note: this is the power dissipated on each half of the disk.
        """
        M = self.binary_mass
        nu = self.viscosity(r)
        sigma = self.surface_density(r, t, torque_effect=torque_effect, isco=isco)
        omega = (G * M / r**3) ** 0.5
        A = -3 / 2 * omega
        D = +1 / 2 * nu * sigma * A**2
        return D

    def temperature(
        self,
        r: float,
        t: float,
        torque_effect: bool = True,
    ) -> float:
        """
        Return the radial surface temperature profile of the disk photosphere

        Note: the viscous power density is defined as the power dissipated on
        each side of the disk.
        """
        D = self.viscous_power_density(r, t, torque_effect=torque_effect)
        return (D / sigma_t) ** (1 / 4)

    def accretion_power(self, t: float) -> float:
        """
        Return the accretion power as a function of time

        The accretion power means the specific orbital energy of gas parcels
        at the ISCO, times the instantaneous accretion rate. In a steady state
        disk, it is twice the disk luminosity.
        """
        return G * self.binary_mass * self.accretion_rate(t) / (2 * self.r_star(t))

    def missing_accretion_power(self, t: float) -> float:
        """
        Return the "missing accretion power"

        The missing accretion power refers to the difference in orbital
        energy, between gas orbiting at the inner edge of the circumbinary
        disk, and gas orbiting at the ISCO. This energy is likely radiated
        above a fraction of a keV. If the mass ratio is low, then much of
        this energy is released from the circum-secondary disk, whose
        characteristic temperature is ~0.3 keV for a 1e5 M_sun secondary. If
        the mass ratio is large then this energy is released from the BH
        minidisks, where the temperature profile is modified by shock
        heating, relative to the temperature profile of a standard
        alpha-disk.

        Following the merger the missing power is zero by definition
        """
        return (
            G
            * self.binary_mass
            * self.accretion_rate(t)
            / 2
            * (-1 / self.r_star(t) + 1 / self.isco_radius)
        )

    def missing_accretion_power_spectrum(self, t: float, nu: float, T: float) -> float:
        """
        Return a BB spectrum with power equal to the missing accretion power
        """
        P = self.missing_accretion_power(t)
        A = P / sigma_t / T**4
        return A * planck_spectrum(nu, T)

    def disk_luminosity(
        self,
        t: float,
        torque_effect: bool = True,
        isco=True,
        disk_extent: float = 1e3,
        epsrel: float = 1e-3,
    ) -> float:
        """
        Return the power radiated from each half of the disk

        In a steady state, the disk luminosity is half the accretion power.
        """
        from scipy.integrate import quad

        def dP_dlogr(logr: float) -> float:
            r = exp(logr)
            return (
                2
                * pi
                * r**2
                * self.viscous_power_density(
                    r, t, torque_effect=torque_effect, isco=isco
                )
            )

        logr0 = log(self.r_star(t))
        logr1 = log(self.r_star(t) * disk_extent)
        return quad(dP_dlogr, logr0, logr1, epsrel=epsrel)[0]

    def spectrum(
        self,
        t: float,
        nu: float,
        torque_effect: bool = True,
        disk_extent: float = 1e3,
        epsrel: float = 1e-3,
    ) -> float:
        """
        Generate the multi-temperature blackbody spectrum

        t: time (negative before merger, positive after)

        nu: photon frequency, Hz

        torque_effect: set to False to disable the torque effect. Disabling
        the torque effect means the binary accretes at the inflow rate, and
        the disk temperature is that of a steady-state disk with the
        instantaneous binary torque.

        disk_extent: ratio of the outer to inner disk radius. It might need to
        be larger tahn 1e3 to probe the emission in the IR, but 1e2 is
        generally far enough out to capture all of the UV.

        epsrel: tolerance passed to scipy.integrate.quad. Don't make it larger
        than 1e-2. Smaller tolerance than 1e-3 will make this function more
        expensive and is probably not justified.
        """
        from scipy.integrate import quad

        def dL_dlogr(logr: float) -> float:
            r = exp(logr)
            T = self.temperature(r, t, torque_effect)
            return 2 * pi * r**2 * planck_spectrum(nu, T)

        logr0 = log(self.r_star(t))
        logr1 = log(self.r_star(t) * disk_extent)
        return quad(dL_dlogr, logr0, logr1, epsrel=epsrel)[0]


def planck_spectrum(nu: float, T: float):
    """
    Return the spectral radiance at frequency nu, integrated over solid angle
    """
    x = clip(h * nu / k / T, None, 709)
    return 2 * pi * h * nu**3 / c**2 / (exp(x) - 1)


def eddington_accretion_rate(mass: float, eta: float = 0.1) -> float:
    """
    Return the Eddington mass accretion rate, for radiative efficiency eta
    """
    L = 1.26e38 * mass / M_sun
    return L / (eta * c**2)
