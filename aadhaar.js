const path = require('path');
const { extractCardDetails } = require('pan-aadhaar-ocr');

const imagePath = path.resolve(__dirname, 'captures', 'original_20260611_183934.jpg');
const requestedCardType = 'AADHAAR';
const cardType = requestedCardType === 'AADHAAR' ? 'AADHAR' : requestedCardType;

extractCardDetails(imagePath, cardType)
    .then((extractedDetails) => {
        console.log(`Aadhaar Number: ${extractedDetails.Number}`);
    })
    .catch((err) => {
        console.error(err);
    });