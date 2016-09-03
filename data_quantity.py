# -*- coding: utf-8 -*-
""" Define each metrix, kinds, units and convertor

The unit tables has been copied from 

[unit of measure](http://unitsofmeasure.org/ucum.html)


"""
from __future__ import division, print_function
__all__ = ["kindofunit", "scaleofunit", "unitofunit", "unitofscale", "unitsofkind", 
           "printofunit", "definitionofunit", "nameofunit", "baseofkind", "isunitof" ,
           "getkinds", "getunits" , "metrixofunit", "baseofunit"
          ]
#####
# Define the colmn indexing for ech kind of table 
#   the  *_N is the number of column 
## columns for Kind table
K_NAME, K_PYTHON, K_DEFINITION, K_BASE, K_ULESS, K_N = range(5+1)
## columns for Unit table
U_NAME,  U_PRINT, U_PYTHON, U_LATEX, U_CI, U_KIND, U_M, U_DIM, U_DEFINITION, U_N =  range(9+1)
## columns of Metrix table 
M_NAME, M_PRINT, M_PYTHON, M_LATEX, M_UC, M_SCALE, M_N = range(6+1)
## columns of Convertor table
C_KINDS, C_TARGETS, C_PYTHON, C_N = range(3+1)


metrix_txt = """
|  name | print | python | latex | u/c | scale  |
|-------|-------|--------|-------|-----|--------|
| yotta | Y     | Y      | Y     | YA  |  1.e24 |
| zetta | Z     | Z      | Z     | ZA  |  1.e21 |
| exa   | E     | E      | E     | EX  |  1.e18 |
| peta  | P     | P      | P     | PT  |  1.e15 |
| tera  | T     | T      | T     | TR  |  1.e12 |
| giga  | G     | G      | G     | GA  |   1.e9 |
| mega  | M     | M      | M     | MA  |   1.e6 |
| kilo  | k     | k      | k     | K   |   1.e3 |
| hecto | h     | h      | h     | H   |   1.e2 |
| deka  | da    | da     | da    | DA  |   1.e1 |
| deci  | d     | d      | d     | D   |  1.e-1 |
| centi | c     | c      | c     | C   |  1.e-2 |
| milli | m     | m      | m     | M   |  1.e-3 |
| micro | μ     | u      | \mu   | U   |  1.e-6 |
| nano  | n     | n      | \eta  | N   |  1.e-9 |
| pico  | p     | p      | p     | P   | 1.e-12 |
| femto | f     | f      | f     | F   | 1.e-15 |
| atto  | a     | a      | a     | A   | 1.e-18 |
| zepto | z     | z      | z     | ZO  | 1.e-21 |
| yocto | y     | y      | y     | YO  | 1.e-24 |
"""



base_kind_txt = """
|        name        | python name | python definition | SI base | unitless |
|--------------------|-------------|-------------------|---------|----------|
| length             | length      |                   | m       | no       |  
| time               | time        |                   | s       | no       |
| mass               | mass        |                   | g       | no       |
| plane angle        | angle       |                   | rad     | yes      |
| temperature        | temperature |                   | K       | no       |
| electric charge    | e_charge    |                   | C       | no       |
| luminous intensity | l_intensity |                   | cd      | no       |
"""

other_kind_txt = """
|            name            |    python name    |     python definition     |  SI base   | unitless |
|----------------------------|-------------------|---------------------------|------------|----------|
| volume                     | volume            | length**3                 | m3         | no       | 
| area                       | area              | length**2                 | m2         | no       |
| lineic number              | lineicnumber      | 1/length                  | 1.0/m      | no       |
| velocity                   | velocity          | length/time               | m/s        | no       |
| acceleration               | acceleration      | length/time**2            | m/s**2     | no       |
| amount of substance        | amount            |                           | n          | no       |
| solid angle                | solid_angle       | angle**2                  | rad**2     | no       |
| frequency                  | frequency         | time**-1                  | Hz         | no       |
| force                      | force             | mass*length/time**2       | N          | no       |
| pressure                   | pressure          | force/length**2           | Pa         | no       |
| energy                     | energy            | force*length              | J          | no       |
| power                      | power             | energy/time               | W          | no       |
| electric current           | e_current         | e_charge/time             | A          | no       |
| electric potential         | e_potential       | energy/e_charge           | V          | no       |
| electric capacitance       | e_capacitance     | e_charge/e_potential      | F          | no       |
| electric resistance        | e_resistance      | e_potential/e_current     | ohm        | no       |
| electric conductance       | e_conductance     | 1./e_resistance           | S          | no       |
| magentic flux              | m_flux            | e_potential*time          | wb         | no       |
| magnetic flux density      | m_flux_density    | m_flux / length**2        | T          | no       |
| magnetic flux intensity    | m_field_intensity | e_current / length        | A/m        | no       |
| magnetic tension           | m_tension         | e_current                 | A          | no       |
| inductance                 | inductance        | m_flux / e_current        | H          | no       |
| luminous flux              | l_flux            | l_intensity * solid_angle | lm         | no       |
| luminous intensity density | brightness        | l_intensity/length**2     | cd/m**2    | no       |
| illuminance                | illuminance       | l_flux/length**2          | lx         | no       |
| action                     | action            | energy*time               | J.s        | no       |
| electric permitivity       | e_permitivity     | e_capacitance/length      | _eps_0     | no       |
| magnetic permeability      | m_permeability    | force/e_current**2        | _mu_0      | no       |
| ion dose                   | ion_dose          | e_charge/mass             | e_charge/g | no       |
| energy dose                | energy_dose       | energy/mass               | J/g        | no       |
| dynamic viscosity          | d_viscosity       | mass/(time*length)        | g/(s*m)    | no       |
| kinematic viscosity        | k_viscosity       | length**2/time            | m**2/s     | no       |
| radioctivity               | radioactivity     | 1/time                    | 1/s        | no       |
| (unclassified)             | unclassified      |                           |            | no       |
| temperature Celcius        | temperature_c     |                           | Cel        | no       |
| fraction                   | fraction          | 1                         | frac       | yes      |
| density                    | density           | amount/volume             | n/m**3     | no       |
| surface density            | s_density         | amount/area               | n/m**2     | no       |
| linear density             | l_density         | amount/length             | n/m        | no       |
"""
## note the fraction is just a dimentionless kind it should be the only one 
## with a definition of 1 



base_units_txt = """
|   name  | print | python | latex | c/i |     kind    |  M  | dim | definition |
|---------|-------|--------|-------|-----|-------------|-----|-----|------------|
| meter   | m     | m      | m     | M   | length      | yes |   1 |        1.0 |
| second  | s     | s      | s     | S   | time        | no  |   1 |        1.0 |
| gram    | g     | g      | g     | G   | mass        | yes |   1 |        1.0 |
| radian  | rad   | rad    | rad   | RAD | angle       | yes |   1 |        1.0 |
| kelvin  | K     | K      | K     | K   | temperature | yes |   1 |        1.0 |
| coulomb | C     | C      | C     | C   | e_charge    | yes |   1 |        1.0 |
| candela | cd    | cd     | cd    | CD  | l_intensity | yes |   1 |        1.0 |
"""

exponant_unit_txt = """
| name | print | python | latex | c/i |  kind  |  M  | dim | definition |
|------|-------|--------|-------|-----|--------|-----|-----|------------|
| m^3  | m3    | m3     | m^3   | m3  | volume | yes |   3 | m          |
| m^2  | m2    | m2     | m^2   | m2  | area   | yes |   2 | m          |
"""

other_units_txt = """
|      name      | print | python | latex | c/i |      kind      |  M  | dim |  definition  |
|----------------|-------|--------|-------|-----|----------------|-----|-----|--------------|
| mole           | mol   | mol    | mol   | MOL | amount         | yes |   1 | 6.0221367e23 |
| steradian      | sr    | sr     | sr    | SR  | solid_angle    | yes |   2 | 1.0          |
| hertz          | Hz    | Hz     | Hz    | HZ  | frequency      | yes |   1 | 1.0          |
| newton         | N     | N      | N     | N   | force          | yes |   1 | kg*m/s**2    |
| pascal         | Pa    | Pa     | Pa    | PAL | pressure       | yes |   1 | N/m**2       |
| joule          | J     | J      | J     | J   | energy         | yes |   1 | N*m          |
| watt           | W     | W      | W     | W   | power          | yes |   1 | J/s          |
| ampere         | A     | A      | A     | A   | e_current      | yes |   1 | C/s          |
| volt           | V     | V      | V     | V   | e_potential    | yes |   1 | J/C          |
| farad          | F     | F      | F     | F   | e_capacitance  | yes |   1 | C/V          |
| ohm            | Ω     | Ohm    | Ohm   | OHM | e_resistance   | yes |   1 | V/A          |
| siemens        | S     | S      | S     | SIE | e_conductance  | yes |   1 | 1.0/Ohm      |
| weber          | Wb    | Wb     | Wb    | WB  | m_flux         | yes |   1 | V*s          |
| tesla          | T     | T      | T     | T   | m_flux_density | yes |   1 | Wb/m**2      |
| henry          | H     | H      | H     | H   | inductance     | yes |   1 | Wb/A         |
| lumen          | lm    | lm     | lm    | LM  | l_flux         | yes |   1 | cd*sr        |
| lux            | lx    | lx     | lx    | LX  | illuminance    | yes |   1 | lm/m**2      |
| degree Celsius | °C    | Cel    | Cel   | CEL | temperature_c  | yes |   1 | 1.0          |
| becquerel      | Bq    | Bq     | Bq    | BQ  | radioactivity  | yes |   1 | 1/s          |
"""

more_units_txt = """
|           name           | print | python |  latex  |  c/i  |   kind   |  M  | dim |  definition unit              |
|--------------------------|-------|--------|---------|-------|----------|-----|-----|-------------------------------|
| degree                   | °     | deg    | deg     | DEG   | angle    | no  |   1 | math.pi/180.                  |
| gon, grade               | g     | gon    | gon     | GON   | angle    | no  |   1 | 0.9*deg                       |
| minute                   | '     | arcmin | \min    | '     | angle    | no  |   1 | deg/60.                       |
| second                   | ''    | arcsec | \second | ''    | angle    | no  |   1 | deg/3600.                     |
| liter                    | l     | l      | l       | L     | volume   | yes |   1 | dm**3                         |
| liter                    | L     | L      | L       | L     | volume   | yes |   1 | l                             |
| are                      | a     | ar     | ar      | AR    | area     | yes |   1 | m**2                          |
| minute                   | min   | min    | min     | MIN   | time     | no  |   1 | 60.                           |
| hour                     | h     | h      | h       | HR    | time     | no  |   1 | 60*min                        |
| day                      | d     | d      | d       | D     | time     | no  |   1 | 24*h                          |
| tropical year            | at    | a_t    | a_t     | ANN_T | time     | no  |   1 | 365.24219 * d                 |
| mean Julian year         | aj    | a_j    | a_j     | ANN_J | time     | no  |   1 | 365.25    * d                 |
| mean Gregorian year      | ag    | a_g    | a_g     | ANN_G | time     | no  |   1 | 365.2425  * d                 |
| year                     | a     | a      | a       | ANN   | time     | no  |   1 | a_j                           |
| week                     | wk    | wk     | wk      | WK    | time     | no  |   1 | 7*d                           |
| synodal month            | mos   | mo_s   | mo_s    | MO_S  | time     | no  |   1 | 29.53059  *d                  |
| mean Julian month        | moj   | mo_j   | mo_j    | MO_J  | time     | no  |   1 | a_j/12                        |
| mean Gregorian month     | mog   | mo_g   | mo_g    | MO_G  | time     | no  |   1 | a_g/12                        |
| month                    | mo    | mo     | mo      | MO    | time     | no  |   1 | mo_j                          |
| tonne                    | t     | t      | t       | TNE   | mass     | yes |   1 | 1e3*kg                        |
| bar                      | bar   | bar    | bar     | BAR   | pressure | yes |   1 | 1e3*Pa                        |
| unified atomic mass unit | u     | u      | u       | AMU   | mass     | yes |   1 | 1.6605402e-24* g              |
| electronvolt             | eV    | eV     | eV      | EV    | energy   | yes |   1 | 1.6021766208e-19*V            |
| astronomic unit          | AU    | AU     | AU      | ASU   | length   | no  |   1 | 149597.870691*Mm              |
| parsec                   | pc    | pc     | pc      | PRS   | length   | yes |   1 | (1.*180/math.pi)*3600.*AU     |
| number                   | #     | n      | #       | #     | amount   | no  |   1 | 1                             | 
"""

dimentionless_units_txt = """
|        name        | print |  python |  latex  |   c/i   |   kind   | M  | dim | definition unit |
|--------------------|-------|---------|---------|---------|----------|----|-----|-----------------|
| unit less number   |       | frac    | frac    | frac    | fraction | no |   1 | 1.0             |
| the number pi      | pi    | pi      | pi      | pi      | fraction | no |   1 | math.pi         |
| percent            | %     | percent | precent | PERCENT | fraction | no |   1 | 1e-2            |
| parts per thousand | ppth  | ppth    | ppth    | PPTH    | fraction | no |   1 | 1e-3            |
| parts per million  | ppm   | ppm     | ppm     | PPM     | fraction | no |   1 | 1e-6            |
| parts per billion  | ppb   | ppb     | ppb     | PPB     | fraction | no |   1 | 1e-9            |
| parts per trillon  | ppt   | ppt     | ppt     | PPT     | fraction | no |   1 | 1e-12           |
"""


# from [table 6](http://unitsofmeasure.org/ucum.html)
natural_units_txt = """
|                name                | print | python |  latex  |   c/i   |      kind      |  M  | dim |      definition value      |
|------------------------------------|-------|--------|---------|---------|----------------|-----|-----|----------------------------|
| velocity of light                  | c     | _c     | c       | [c]     | velocity       | yes |   1 | 299792458*m/s              |
| Planck  constant                   | h     | _h     | [h]     | [H]     | action         | yes |   1 | 6.6260755e-34   *J*s       |
| Boltzmann constant                 | k     | _k     | [k]     | [K]     | unclassified   | yes |   1 | 1.380658e-23    *J/K       |
| permittivity of vacuum             | ε0    | _eps_0 | [eps_0] | [EPS_0] | e_permitivity  | yes |   1 | 8.854187817e-12 *F/m       |
| permeability of vacuum             | μ0    | _mu_0  | [mu_0]  | [MU_0]  | m_permeability | yes |   1 | 4*math.pi*1e-7  *N/A**2    |
| elementary charge                  | e     | _e     | [e]     | [E]     | e_charge       | yes |   1 | 1.60217733e-19  *C         |
| electron mass                      | me    | _m_e   | [m_e]   | [M_E]   | mass           | yes |   1 | 9.1093897e-28   *g         |
| proton mass                        | mp    | _m_p   | [m_p]   | [M_P]   | mass           | yes |   1 | 1.6726231e-24   *g         |
| Newtonian constant of gravitation  | G     | _G     | [G]     | [GC]    | unclassified   | yes |   1 | 6.67259e-11 * m**3/kg/s**2 |
| standard acceleration of free fall | gn    | _g     | [g]     | [G]     | acceleration   | yes |   1 | 9.80665  * m/s**2          |
| standard atmosphere                | atm   | _atm   | atm     | ATM     | pressure       | no  |   1 | 101325  * Pa               |
| light-year                         | l.y.  | _ly    | [ly]    | [LY]    | length         | yes |   1 | _c*a_j                     |
| gram-force                         | gf    | _gf    | gf      | GF      | force          | yes |   1 | g*_g                       |
"""
# supressed
"""
| pound force                        | lbf   | _lbf_av | [lbf_av] | [LBF_AV] | force          | no  | _lb_av*_g                  |
"""

cgs_units_txt = """
|                 name                 | print | python | latex |  c/i  |        kind       |  M  | dim |     definition    |
|--------------------------------------|-------|--------|-------|-------|-------------------|-----|-----|-------------------|
| Kayser                               | K     | Ky     | Ky    | KY    | lineicnumber      | yes |   1 | cm**-1            |
| Gal                                  | Gal   | Gal    | Gal   | GL    | acceleration      | yes |   1 | cm/s**2           |
| dyne                                 | dyn   | dyn    | dyn   | DYN   | force             | yes |   1 | g*cm/s**2         |
| erg                                  | erg   | erg    | erg   | ERG   | energy            | yes |   1 | dyn*cm            |
| Poise                                | P     | P      | P     | P     | d_viscosity       | yes |   1 | dyn*s/cm**2       |
| Biot                                 | Bi    | Bi     | Bi    | BI    | e_current         | yes |   1 | 10*A              |
| Stokes                               | St    | St     | St    | ST    | k_viscosity       | yes |   1 | cm**2/s           |
| Maxwell (flux of magnetic induction) | Mx    | Mx     | Mx    | MX    | m_flux            | yes |   1 | 1e-8*Wb           |
| Gauss                                | Gs,G  | G      | G     | GS    | m_flux_density    | yes |   1 | 1e-4*T            |
| Oersted                              | Oe    | Oe     | Oe    | OE    | m_field_intensity | yes |   1 | 250/math.pi*A/m   |
| Gilbert                              | Gb    | Gb     | Gb    | GB    | m_tension         | yes |   1 | Oe*cm             |
| stilb                                | sb    | sb     | sb    | SB    | brightness        | yes |   1 | cd/cm**2          |
| Lambert                              | L     | Lmb    | Lmb   | LMB   | brightness        | yes |   1 | cd/cm**2/math.pi  |
| phot                                 | ph    | ph     | ph    | PHT   | illuminance       | yes |   1 | 1e-4 *lx          |
| Curie                                | Ci    | Ci     | Ci    | CI    | radioactivity     | yes |   1 | 3.7e10*  Bq       |
| Roentgen                             | R     | R      | R     | ROE   | ion_dose          | yes |   1 | 2.58e-4  *   C/kg |
| radiation  absorbed dose             | RAD   | RAD    | RAD   | [RAD] | energy_dose       | yes |   1 | 100 * erg/g       |
| radiation  equivalent man            | REM   | REM    | REM   | [REM] | energy_dose       | yes |   1 | RAD               |
"""

##
# unified U.S and British Imperial units
# table 8 
international_units_txt ="""
|       name      |  print  | python |  latex  |   c/i   |   kind   | M  | dim |     definition     |
|-----------------|---------|--------|---------|---------|----------|----|-----|--------------------|
| inch          I | [in_i]  | in_i   | [in_i]  | [IN_I]  | length   | no |   1 | 2.54*cm            |
| foot          I | [ft_i]  | ft_i   | [ft_i]  | [FT_I]  | length   | no |   1 | 12*in_i            |
| yard          I | [yd_i]  | yd_i   | [yd_i]  | [YD_I]  | length   | no |   1 | 3*ft_i             |
| mile          I | [mi_i]  | mi_i   | [mi_i]  | [MI_I]  | length   | no |   1 | 5280*ft_i          |
| fathom        I | [fth_i] | fth_i  | [fth_i] | [FTH_I] | length   | no |   1 | 6 * ft_i           |
| nautical mile I | [nmi_i] | nmi_i  | [nmi_i] | [NMI_I] | length   | no |   1 | 1852 *m            |
| knot          I | [kn_i]  | kn_i   | [kn_i]  | [KN_I]  | velocity | no |   1 | nmi_i/h            |
| square inch   I | [sin_i] | sin_i  | [sin_i] | [SIN_I] | area     | no |   2 | in_i**2            |
| square foot   I | [sft_i] | sft_i  | [sft_i] | [SFT_I] | area     | no |   2 | ft_i**2            |
| square yard   I | [syd_i] | syd_i  | [syd_i] | [SYD_I] | area     | no |   2 | yd_i**2            |
| cubic inch    I | [cin_i] | cin_i  | [cin_i] | [CIN_I] | volume   | no |   3 | in_i**3            |
| cubic foot    I | [cft_i] | cft_i  | [cft_i] | [CFT_I] | volume   | no |   3 | ft_i**3            |
| cubic yard    I | [cyd_i] | cyd_i  | [cyd_i] | [CYD_I] | volume   | no |   3 | yd_i**3            |
| board foot    I | [bf_i]  | bf_i   | [bf_i]  | [BF_I]  | volume   | no |   3 | 144 * in_i**3      |
| cord          I | [cr_i]  | cr_i   | [cr_i]  | [CR_I]  | volume   | no |   3 | 128 * ft_i**3      |
| mil           I | [mil_i] | mil_i  | [mil_i] | [MIL_I] | length   | no |   1 | 1.e-3 *in_i        |
| circular  mil I | [cml_i] | cml_i  | [cml_i] | [CML_I] | area     | no |   2 | math.pi/4*mil_i**2 |
| hand          I | [hd_i]  | hd_i   | [hd_i]  | [HD_I]  | length   | no |   1 | 4*in_i             |
"""

##Table 9: Older U.S. “survey” lengths (also called "statute" lengths)
old_us_units_txt = """
|                 name                |  print   | python |  latex   |   c/i    |  kind  | M  | dim |                  |
|-------------------------------------|----------|--------|----------|----------|--------|----|-----|------------------|
| foot US                             | [ft_us]  | ft_us  | [ft_us]  | [FT_US]  | length | no |   1 | 1200 *m/3937.    |
| yard US                             | [yd_us]  | yd_us  | [yd_us]  | [YD_US]  | length | no |   1 | 3 *ft_us         |
| inch US                             | [in_us]  | in_us  | [in_us]  | [IN_US]  | length | no |   1 | 1  *ft_us/12.    |
| rod  US                             | [rd_us]  | rd_us  | [rd_us]  | [RD_US]  | length | no |   1 | 16.5 *ft_us      |
| Gunter's chain, Surveyor's chain US | [ch_us]  | ch_us  | [ch_us]  | [CH_US]  | length | no |   1 | 4  *rd_us        |
| link for Gunter's chain          US | [lk_us]  | lk_us  | [lk_us]  | [LK_US]  | length | no |   1 | 1  *ch_us/100    |
| Ramden's chain, Engineer's chain US | [rch_us] | rch_us | [rch_us] | [RCH_US] | length | no |   1 | 100  *ft_us      |
| link for Ramden's chain          US | [rlk_us] | rlk_us | [rlk_us] | [RLK_US] | length | no |   1 | 1   * rch_us/100 |
| fathom  US                          | [fth_us] | fth_us | [fth_us] | [FTH_US] | length | no |   1 | 6   * ft_us      |
| furlong US                          | [fur_us] | fur_us | [fur_us] | [FUR_US] | length | no |   1 | 40  * rd_us      |
| mile    US                          | [mi_us]  | mi_us  | [mi_us]  | [MI_US]  | length | no |   1 | 8   * fur_us     |
| acre    US                          | [acr_us] | acr_us | [acr_us] | [ACR_US] | area   | no |   2 | 160 *rd_us**2    |
| square rod  US                      | [srd_us] | srd_us | [srd_us] | [SRD_US] | area   | no |   2 | 1   * rd_us**2   |
| square mile US                      | [smi_us] | smi_us | [smi_us] | [SMI_US] | area   | no |   2 | 1   * mi_us**2   |
| section     US                      | [sct]    | sct    | [sct]    | [SCT]    | area   | no |   2 | 1   * mi_us**2   |
| township    US                      | [twp]    | twp    | [twp]    | [TWP]    | area   | no |   2 | 36  * sct        |
| mil         US                      | [mil_us] | mil_us | [mil_us] | [MIL_US] | length | no |   1 | 1.e-3 *in_us     |
"""

british_imperial_txt= """
|           name          |  print   | python |  latex   |   c/i    |   kind   | M  | dim |    definition    |
|-------------------------|----------|--------|----------|----------|----------|----|-----|------------------|
| inch                    | [in_br]  | in_br  | [in_br]  | [IN_BR]  | length   | no |   1 | 2.539998 * cm    |
| foot                    | [ft_br]  | ft_br  | [ft_br]  | [FT_BR]  | length   | no |   1 | 12  *in_br       |
| rod                     | [rd_br]  | rd_br  | [rd_br]  | [RD_BR]  | length   | no |   1 | 16.5  *ft_br     |
| Gunter's chain          | [ch_br]  | ch_br  | [ch_br]  | [CH_BR]  | length   | no |   1 | 4  *rd_br        |
| link for Gunter's chain | [lk_br]  | lk_br  | [lk_br]  | [LK_BR]  | length   | no |   1 | 1  *ch_br/100.   |
| fathom                  | [fth_br] | fth_br | [fth_br] | [FTH_BR] | length   | no |   1 | 6  *ft_br        |
| pace                    | [pc_br]  | pc_br  | [pc_br]  | [PC_BR]  | length   | no |   1 | 2.5  *ft_br      |
| yard                    | [yd_br]  | yd_br  | [yd_br]  | [YD_BR]  | length   | no |   1 | 3   *ft_br       |
| mile                    | [mi_br]  | mi_br  | [mi_br]  | [MI_BR]  | length   | no |   1 | 5280  *ft_br     |
| nautical mile           | [nmi_br] | nmi_br | [nmi_br] | [NMI_BR] | length   | no |   1 | 6080  *ft_br     |
| knot                    | [kn_br]  | kn_br  | [kn_br]  | [KN_BR]  | velocity | no |   1 | 1  *nmi_br/h     |
| acre                    | [acr_br] | acr_br | [acr_br] | [ACR_BR] | area     | no |   1 | 4840 *yd_br**2 |
"""



###
# The python convertor is a function accepting one argument, the quentity value
# Becarefull when setting the function not to make infinite convertion loop
#
base_convertors_txt = """
|      kind     |  kind targets |                  python                  |
|---------------|---------------|------------------------------------------|
| temperature   | temperature_c | lambda tk:  tk.to('K')._value - 273.15   |
| temperature_c | temperature   | lambda tc:  tc.to('Cel')._value + 273.15 |
| *             | fraction      | lambda x: x                              |
"""
base_convertors_txt = """
|      kind     |  kind targets |                  python                           |
|---------------|---------------|---------------------------------------------------|
| temperature   | temperature_c | lambda tk,u:  (convert(tk,u,'K') - 273.15, 'Cel') |
| temperature_c | temperature   | lambda tc,u:  (convert(tc,u,'Cel') + 273.15, 'K') |
| *             | fraction      | lambda x,u:   (x, 'frac')                         |
| fraction      | angle         | lambda x,u:  (convert(x, u,'frac'), 'rad')        |
"""


unit_txt_tables = [base_units_txt, exponant_unit_txt, other_units_txt, dimentionless_units_txt, 
                more_units_txt,natural_units_txt, cgs_units_txt, 
                international_units_txt, old_us_units_txt, british_imperial_txt
                ]
kind_txt_tables = [base_kind_txt, other_kind_txt]
metrix_txt_tables =[metrix_txt]
convertor_txt_tables = [base_convertors_txt]


