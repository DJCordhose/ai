// adapted from
// http://www.html5canvastutorials.com/labs/html5-canvas-graphing-an-equation/

class Plot {
    constructor(config) {
        // user defined properties
        this.canvas = config.canvas;
        this.minX = config.minX;
        this.minY = config.minY;
        this.maxX = config.maxX;
        this.maxY = config.maxY;
        this.unitsPerTickX = config.unitsPerTickX;
        this.unitsPerTickY = config.unitsPerTickY;

        // constants
        this.axisColor = '#aaa';
        this.font = '8pt Calibri';
        this.tickSize = 10;

        // relationships
        this.context = this.canvas.getContext('2d');
        this.rangeX = this.maxX - this.minX;
        this.rangeY = this.maxY - this.minY;
        this.unitX = this.canvas.width / this.rangeX;
        this.unitY = this.canvas.height / this.rangeY;
        this.centerY = Math.round(Math.abs(this.minY / this.rangeY) * this.canvas.height);
        this.centerX = Math.round(Math.abs(this.minX / this.rangeX) * this.canvas.width);
        this.iteration = (this.maxX - this.minX) / 1000;
        this.scaleX = this.canvas.width / this.rangeX;
        this.scaleY = this.canvas.height / this.rangeY;

        // draw x and y axis
        this.drawXAxis();
        this.drawYAxis();
    }

    samplePositions() {
        const samples = [];
        for (let x = this.minX + this.iteration; x <= this.maxX; x += this.iteration) {
            samples.push(x);
        }
        return samples;
    }

    clear() {
        const { canvas, context } = this;
        // Store the current transformation matrix
        context.save();

        // Use the identity matrix while clearing the canvas
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, canvas.width, canvas.height);

        // Restore the transform
        context.restore();
        this.drawXAxis();
        this.drawYAxis();
    }

    drawXAxis() {
        const context = this.context;
        const unitsPerTick = this.unitsPerTickX;
        context.save();
        context.beginPath();
        context.moveTo(0, this.centerY);
        context.lineTo(this.canvas.width, this.centerY);
        context.strokeStyle = this.axisColor;
        context.lineWidth = 1;
        context.stroke();
        // draw tick marks
        const xPosIncrement = unitsPerTick * this.unitX;
        let xPos, unit;
        context.font = this.font;
        context.textAlign = 'center';
        context.textBaseline = 'top';

        // draw left tick marks
        xPos = this.centerX - xPosIncrement;
        unit = -1 * unitsPerTick;
        while (xPos > 0) {
            context.moveTo(xPos, this.centerY - this.tickSize / 2);
            context.lineTo(xPos, this.centerY + this.tickSize / 2);
            context.stroke();
            context.fillText(unit, xPos, this.centerY + this.tickSize / 2 + 3);
            unit -= unitsPerTick;
            xPos = Math.round(xPos - xPosIncrement);
        }

        // draw right tick marks
        xPos = this.centerX + xPosIncrement;
        unit = unitsPerTick;
        while (xPos < this.canvas.width) {
            context.moveTo(xPos, this.centerY - this.tickSize / 2);
            context.lineTo(xPos, this.centerY + this.tickSize / 2);
            context.stroke();
            context.fillText(unit, xPos, this.centerY + this.tickSize / 2 + 3);
            unit += unitsPerTick;
            xPos = Math.round(xPos + xPosIncrement);
        }
        context.restore();
    }

    drawYAxis() {
        const context = this.context;
        const unitsPerTick = this.unitsPerTickY;
        context.save();
        context.beginPath();
        context.moveTo(this.centerX, 0);
        context.lineTo(this.centerX, this.canvas.height);
        context.strokeStyle = this.axisColor;
        context.lineWidth = 2;
        context.stroke();

        // draw tick marks
        const yPosIncrement = unitsPerTick * this.unitY;
        let yPos, unit;
        context.font = this.font;
        context.textAlign = 'right';
        context.textBaseline = 'middle';

        // draw top tick marks
        yPos = this.centerY - yPosIncrement; unit = unitsPerTick;
        while (yPos > 0) {
            context.moveTo(this.centerX - this.tickSize / 2, yPos);
            context.lineTo(this.centerX + this.tickSize / 2, yPos);
            context.stroke();
            context.fillText(unit, this.centerX - this.tickSize / 2 - 3, yPos);
            unit += unitsPerTick;
            yPos = Math.round(yPos - yPosIncrement);
        }

        // draw bottom tick marks
        yPos = this.centerY + yPosIncrement;
        unit = -1 * unitsPerTick;
        while (yPos < this.canvas.height) {
            context.moveTo(this.centerX - this.tickSize / 2, yPos);
            context.lineTo(this.centerX + this.tickSize / 2, yPos);
            context.stroke();
            context.fillText(unit, this.centerX - this.tickSize / 2 - 3, yPos);
            unit -= unitsPerTick;
            yPos = Math.round(yPos + yPosIncrement);
        }
        context.restore();
    }

    drawEquation(equation, color, thickness) {
        const context = this.context;
        context.save();
        context.save();
        this.transformContext();

        context.beginPath();
        context.moveTo(this.minX, equation(this.minX));

        for (let x = this.minX + this.iteration; x <= this.maxX; x += this.iteration) {
            context.lineTo(x, equation(x));
        }
        context.restore();
        context.lineJoin = 'round';
        context.lineWidth = thickness;
        context.strokeStyle = color;
        context.stroke();
        context.restore();
    }

    drawPoints(points, color, thickness) {
        const context = this.context;
        context.save();
        context.save();
        this.transformContext();

        context.beginPath();

        const [beginX, beginY] = points[0]; 
        context.moveTo(beginX, beginY);

        for (const [x, y] of points) {
            context.lineTo(x, y);
        }

        context.restore();
        context.lineJoin = 'round';
        context.lineWidth = thickness;
        context.strokeStyle = color;
        context.stroke();
        context.restore();
    }

    scatterPlot(points, color, radius) {
        const context = this.context;
        context.save();
        this.transformContext();

        context.fillStyle = color;

        for (const [x, y] of points) {
            context.beginPath();
            // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/ellipse
            context.ellipse(x, y, radius/this.scaleX, radius/this.scaleY, 0, 0, 2 * Math.PI);
            context.fill();
        }
        context.restore();
    }

    transformContext() {
        // move context to center of canvas
        this.context.translate(this.centerX, this.centerY);

        /*
         * stretch grid to fit the canvas window, and
         * invert the y scale so that that increments
         * as you move upwards
         */
        this.context.scale(this.scaleX, -this.scaleY);
    }
}