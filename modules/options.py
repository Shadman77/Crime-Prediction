import numpy


def select_weapon(X):
    inp = input("Enter the weapon: ")
    weapons = ["Blunt Object",
               "Drowning", "Drugs", "Explosives",
               "Fall", "Fire", "Firearm", "Gun", "Handgun",
               "Knife", "Poison", "Rifle", "Shotgun", "Strangulation", "Suffocation", "Unknown"]

    for weapon in weapons:
        if weapon.lower() == inp.lower():
            X.at[0, "Weapon_" + weapon] = numpy.int64(1)
        else:
            X.at[0, "Weapon_" + weapon] = numpy.int64(0)

        print("Weapon_" + weapon,
              X.at[0, "Weapon_" + weapon], type(X.at[0, "Weapon_" + weapon]))

    return X

def select_ethnicity(X):
    inp = input("Enter the ethnicity: ")
    ethnicities = ["Hispanic", "Not Hispanic", "Unknown"]

    for ethnicity in ethnicities:
        if ethnicity.lower() == inp.lower():
            X.at[0, "Perp_Ethnicity_" + ethnicity] = numpy.int64(1)
        else:
            X.at[0, "Perp_Ethnicity_" + ethnicity] = numpy.int64(0)

        print("Perp_Ethnicity_" + ethnicity,
              X.at[0, "Perp_Ethnicity_" + ethnicity], type(X.at[0, "Perp_Ethnicity_" + ethnicity]))

    return X

def select_race(X):
    inp = input("Enter the race: ")
    races = ["Asian/Pacific Islander", "Black", "Native American/Alaska Native", "Unknown", 
    "White"]

    for race in races:
        if race.lower() == inp.lower():
            X.at[0, "Perp_Race_" + race] = numpy.int64(1)
        else:
            X.at[0, "Perp_Race_" + race] = numpy.int64(0)

        print("Perp_Race_" + race,
              X.at[0, "Perp_Race_" + race], type(X.at[0, "Perp_Race_" + race]))

    return X


def select_loc_desc(X):
    inp = input("Enter the location description: ")
    locs = ["ABANDONED BUILDING", "AIRCRAFT", "AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA",
               "AIRPORT BUILDING NON-TERMINAL - SECURE AREA", "AIRPORT EXTERIOR - NON-SECURE AREA", "AIRPORT EXTERIOR - SECURE AREA",
               "AIRPORT PARKING LOT", "AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA", "AIRPORT TERMINAL LOWER LEVEL - SECURE AREA",
               "AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA", "AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA",
               "AIRPORT TERMINAL UPPER LEVEL - SECURE AREA", "AIRPORT TRANSPORTATION SYSTEM (ATS)", "AIRPORT VENDING ESTABLISHMENT",
               "AIRPORT/AIRCRAFT", "ALLEY", "ANIMAL HOSPITAL", "APARTMENT", "APPLIANCE STORE", "ATHLETIC CLUB",
               "ATM (AUTOMATIC TELLER MACHINE)", "AUTO", "BANK", "BAR OR TAVERN", "BARBER SHOP/BEAUTY SALON",
               "BARBERSHOP", "BOAT/WATERCRAFT", "BOWLING ALLEY", "BRIDGE", "CAR WASH", "CEMETARY", "CHA APARTMENT",
               "CHA HALLWAY/STAIRWELL/ELEVATOR", "CHA PARKING LOT/GROUNDS", "CHURCH/SYNAGOGUE/PLACE OF WORSHIP",
               "CLEANING STORE", "COIN OPERATED MACHINE", "COLLEGE/UNIVERSITY GROUNDS", "COLLEGE/UNIVERSITY RESIDENCE HALL",
               "COMMERCIAL / BUSINESS OFFICE", "CONSTRUCTION SITE", "CONVENIENCE STORE", "CREDIT UNION",
               "CTA BUS", "CTA BUS STOP", "CTA GARAGE / OTHER PROPERTY", "CTA PLATFORM", "CTA STATION",
               "CTA TRACKS - RIGHT OF WAY", "CTA TRAIN", "CURRENCY EXCHANGE", "DAY CARE CENTER", "DELIVERY TRUCK",
               "DEPARTMENT STORE", "DRIVEWAY - RESIDENTIAL", "DRUG STORE", "FACTORY/MANUFACTURING BUILDING", "FEDERAL BUILDING",
               "FIRE STATION", "FOREST PRESERVE", "GAS STATION", "GOVERNMENT BUILDING/PROPERTY", "GROCERY FOOD STORE",
               "HIGHWAY/EXPRESSWAY", "HOSPITAL BUILDING/GROUNDS", "HOTEL/MOTEL", "HOUSE", "JAIL / LOCK-UP FACILITY",
               "LAKEFRONT/WATERFRONT/RIVERBANK", "LIBRARY", "MEDICAL/DENTAL OFFICE", "MOVIE HOUSE/THEATER",
               "NEWSSTAND", "NURSING HOME/RETIREMENT HOME", "OTHER", "OTHER COMMERCIAL TRANSPORTATION",
               "OTHER RAILROAD PROP / TRAIN DEPOT", "PARK PROPERTY", "PARKING LOT", "PARKING LOT/GARAGE(NON.RESID.)",
               "PAWN SHOP", "POLICE FACILITY/VEH PARKING LOT", "POOL ROOM", "PORCH", "RESIDENCE", "RESIDENCE PORCH/HALLWAY",
               "RESIDENCE-GARAGE", "RESIDENTIAL YARD (FRONT/BACK)", "RESTAURANT", "SAVINGS AND LOAN", "SCHOOL, PRIVATE, BUILDING",
               "SCHOOL, PRIVATE, GROUNDS", "SCHOOL, PUBLIC, BUILDING", "SCHOOL, PUBLIC, GROUNDS", "SIDEWALK",
               "SMALL RETAIL STORE", "SPORTS ARENA/STADIUM", "STREET", "TAVERN/LIQUOR STORE", "TAXICAB", "VACANT LOT/LAND",
               "VEHICLE - DELIVERY TRUCK", "VEHICLE - OTHER RIDE SERVICE", "VEHICLE NON-COMMERCIAL", "VEHICLE-COMMERCIAL",
               "WAREHOUSE", "YARD"]

    for loc in locs:
        if loc.lower() == inp.lower():
            X.at[0, "Loc_Dec_" + loc] = numpy.int64(1)
        else:
            X.at[0, "Loc_Dec_" + loc] = numpy.int64(0)

        print("Loc_Dec_" + loc,
              X.at[0, "Loc_Dec_" + loc], type(X.at[0, "Loc_Dec_" + loc]))

    return X

def select_sex(X):
    inp = input("Enter the sex:")
    if inp.lower() == "male":
        X.at[0, "Perpetrator Sex"] = numpy.int64(1)
    else:
        X.at[0, "Perpetrator Sex"] = numpy.int64(0)
    print("Perpetrator Sex",
              X.at[0, "Perpetrator Sex"], type(X.at[0, "Perpetrator Sex"]))
    return X
    
