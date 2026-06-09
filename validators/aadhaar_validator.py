import re
from datetime import datetime


class AadhaarValidator:

    def validate_aadhaar(self, aadhaar):

        if not aadhaar:
            return False

        aadhaar = aadhaar.replace(" ", "")

        if not aadhaar.isdigit():
            return False

        if len(aadhaar) != 12:
            return False

        return True

    def validate_dob(self, dob):

        if not dob:
            return False

        try:

            dob_date = datetime.strptime(
                dob,
                "%d/%m/%Y"
            )

            if dob_date > datetime.now():
                return False

            age = (
                datetime.now() - dob_date
            ).days / 365

            if age < 0:
                return False

            if age > 120:
                return False

            return True

        except:
            return False

    def validate_mobile(self, mobile):

        if not mobile:
            return False

        return bool(
            re.match(
                r"^[6-9]\d{9}$",
                mobile
            )
        )

    def validate_name(self, name):

        if not name:
            return False

        if len(name.strip()) < 3:
            return False

        if re.search(r"\d", name):
            return False

        return True

    def find_missing_fields(self, data):

        missing = []

        required_fields = [
            "name",
            "dob",
            "gender",
            "mobile",
            "aadhaar_number"
        ]

        for field in required_fields:

            value = data.get(field)

            if value is None:
                missing.append(field)

            elif isinstance(value, str):

                if value.strip() == "":
                    missing.append(field)

        return missing

    def validate(self, data):

        result = {}

        result["name_valid"] = self.validate_name(
            data.get("name")
        )

        result["dob_valid"] = self.validate_dob(
            data.get("dob")
        )

        result["mobile_valid"] = self.validate_mobile(
            data.get("mobile")
        )

        result["aadhaar_valid"] = self.validate_aadhaar(
            data.get("aadhaar_number")
        )

        result["missing_fields"] = self.find_missing_fields(
            data
        )

        result["overall_status"] = "VALID"

        if (
            not result["name_valid"]
            or not result["dob_valid"]
            or not result["mobile_valid"]
            or not result["aadhaar_valid"]
            or len(result["missing_fields"]) > 0
        ):
            result["overall_status"] = "INVALID"

        return result