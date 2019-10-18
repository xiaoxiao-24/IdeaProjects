package PropertySale

import java.sql.Timestamp

case class PropertySaleClass(
                              ParId: String,
                              HouseNum: Integer,
                              Fraction: String,
                              AddressDir: String,
                              Street: String,
                              AddressSuf: String,
                              UnitDesc: String,
                              UnitNo: String,
                              City: String,
                              State: String,
                              ZipCode: String,
                              SchoolCode: Integer,
                              SchoolDesc: String,
                              MuniCode: Integer,
                              MuniDesc: String,
                              RecodeDate: Timestamp,
                              SaleDate: Timestamp,
                              Price: Integer,
                              DeedBook: Integer,
                              DeedPage: Integer,
                              SaleCode: String,
                              SaleDesc: String,
                              Instrtyp: String,
                              InstrtypDesc: String
                            )
