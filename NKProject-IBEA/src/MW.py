import numpy
import scipy
import functools
from operator import itemgetter
from operator import attrgetter


def biased(m,gene):
    return 1+numpy.sum(biasedz(m,gene))

def biasedz(m,gene):
    zs = [1-numpy.exp(-10*numpy.power((numpy.power(gene[i],len(gene)-m))-0.5-(i/(2*len(gene))),2)) for i in range(m, len(gene))]
    return zs

def multimodal(m,gene):
    return 1+numpy.sum(multimodalz(m,gene))

def multimodalz(m,gene):
    zs = [1.5+0.1*(1-numpy.exp(-10*numpy.power(gene[i]-i/len(gene),2))-1.5*numpy.cos(2*numpy.pi*(1-numpy.exp(-10*numpy.power(gene[i]-i/len(gene),2))))/(len(gene))) for i in range(m, len(gene))]
    return zs

def variablelinkage(m,gene):
    return 1+numpy.sum(biasedz(m,gene))

def variablelinkagez(m,gene):
    zs = [2*numpy.power(gene[i]+numpy.power(gene[i-1]-0.5,2)-1,2) for i in range(m, len(gene))]
    return zs

def plainf1(gene):
    return gene[0]

def MW1(m,gene):
    f1 = plainf1(gene)
    f2 = biased(m,gene) * (1-(0.85*f1/biased(m,gene)))
    l = numpy.sqrt(2)*(f2-f1)
    c = 1-f1-f2+0.5*numpy.power(numpy.sin(2*numpy.pi*l),8)
    return [f1,f2],[c]

def MW2(m,gene):
    f1 = plainf1(gene)
    f2 = multimodal(m,gene) * (1-(f1/multimodal(m,gene)))
    l = numpy.sqrt(2)*(f2-f1)
    c = 1-f1-f2+0.5*numpy.power(numpy.sin(3*numpy.pi*l),8)
    return [f1,f2],[c]

def MW3(m,gene):
    f1 = plainf1(gene)
    f2 = variablelinkage(m,gene) * (1-(f1/variablelinkage(m,gene)))
    l = numpy.sqrt(2)*(f2-f1)
    c1 = 1.05 - f1 - f2 + 0.35 * numpy.power(numpy.sin(3*numpy.pi*l),8)
    c2 = -(0.85 - f1 - f2 +0.3*numpy.power(numpy.sin(3*numpy.pi*l),8))
    return [f1,f2],[c1,c2]

def MW4(m,gene):
    fbase = numpy.ones(m-1) - numpy.array(gene[:m-1])
    f = []
    f.append(biased(m,gene) * numpy.prod(fbase))
    for i in range(1,m-1):
        fkbase = numpy.ones(m - i) - numpy.array(gene[:m-i])
        f.append(biased(m,gene)*gene[m - i + 1]*numpy.prod(fkbase))
    f.append(biased(m,gene) * plainf1(gene))
    l = f[-1] - numpy.sum(numpy.array(f[:-1]))
    c = 1 + 0.4*numpy.power(numpy.sin(2.5+numpy.pi*l),8)-numpy.sum(numpy.array(f))
    return f,[c]

def MW5(m,gene):
    f1 = biased(m,gene) * plainf1(gene)
    f2 = biased(m,gene) * numpy.sqrt(1-(f1/biased(m,gene)))
    l1 = numpy.arctan(f2/(f1+1e-12))
    l2 = 0.5 * numpy.pi - 2 * numpy.abs(l1-0.25*numpy.pi)
    c1 = numpy.power(1.7-0.2*numpy.sin(2*l1), 2) - numpy.power(f1,2)-numpy.power(f2,2)
    c2 = - (numpy.power(1+0.5*numpy.sin(6*numpy.power(l2,3)), 2) - numpy.power(f1,2)-numpy.power(f2,2))
    c3 = - (numpy.power(1 - 0.45 * numpy.sin(6 * numpy.power(l2, 3)), 2) - numpy.power(f1, 2) - numpy.power(f2, 2))
    return [f1,f2],[c1,c2,c3]

def MW6(m,gene):
    f1 = multimodal(m,gene) * plainf1(gene)
    f2 = multimodal(m,gene) * numpy.sqrt(numpy.power(1.1,2)-numpy.power(f1/multimodal(m,gene),2))
    l = numpy.power(numpy.cos(6*numpy.arctan(f2/(f1+1e-12))),10)
    c = 1 - numpy.power((f1/(1+0.15 * l)), 2) -numpy.power((f2/(1+0.75 * l)), 2)
    return [f1,f2], [c]

def MW7(m,gene):
    f1 = variablelinkage(m,gene) * plainf1(gene)
    f2 = variablelinkage(m,gene) * numpy.sqrt(1-numpy.power(f1/variablelinkage(m,gene),2))
    l = numpy.arctan(f2 / (f1 + 1e-12))
    c1 = numpy.power(1.2 + 0.4 * numpy.power(numpy.sin(4 * l),16), 2) - numpy.power(f1, 2) - numpy.power(f2, 2)
    c2 = -(numpy.power(1.15 - 0.2 * numpy.power(numpy.sin(4 * l),8), 2) - numpy.power(f1, 2) - numpy.power(f2, 2))
    return [f1,f2],[c1,c2]

def MW8(m,gene):
    fbase = numpy.array(gene[:m-1])
    fbase = 0.5 * numpy.pi * fbase
    fbase = numpy.cos(fbase)
    f = []
    f.append(multimodal(m,gene) * numpy.prod(fbase))
    if m > 3:
        for i in range(2, m - 1):
            fkbase = fbase.tolist()[:m - i]
            f.append(multimodal(m, gene) * numpy.sin(0.5 * numpy.pi * gene[m - i + 1]) * numpy.prod(fkbase))
    else:
        fkbase = fbase.tolist()[:m - 2]
        f.append(multimodal(m, gene) * numpy.sin(0.5 * numpy.pi * gene[m - 2 + 1]) * numpy.prod(fkbase))
    f.append(multimodal(m,gene)*numpy.sin(0.5 * numpy.pi * plainf1(gene)))
    sf = numpy.power(f,2)
    l = numpy.arcsin(f[-1]/numpy.sqrt(numpy.sum(sf)))
    c = numpy.power(1.25-0.5*numpy.power(numpy.sin(6*l),2),2)-numpy.sum(f)
    return f,[c]

def MW9(m,gene):
    f1 = biased(m,gene) * plainf1(gene)
    f2 = biased(m,gene) * (1-numpy.power(f1/biased(m,gene),2))
    T1 = (1-0.64*numpy.power(f1,2)-f2)*(1-0.36*numpy.power(f1,2)-f2)
    T2 = numpy.power(1.35,2)-numpy.power(f1+0.35,2)-f2
    T3 = numpy.power(1.15, 2) - numpy.power(f1 + 0.15, 2) - f2
    c = numpy.max(numpy.array([-T1,-(T2*T3)]))
    return [f1,f2],[c]

def MW10(m,gene):
    f1 = multimodal(m,gene) * numpy.power(plainf1(gene),len(gene))
    f2 = multimodal(m,gene) * (1-numpy.power(f1/multimodal(m,gene),2))
    c1 = (2-4*numpy.power(f1,2)-f2)*(2-8*numpy.power(f1,2)-f2)
    c2 = -(2-2*numpy.power(f1,2)-f2)*(2-16*numpy.power(f1,2)-f2)
    c3 = -(1-numpy.power(f1,2)-f2)*(1.2-1.2*numpy.power(f1,2)-f2)
    return [f1,f2],[c1,c2,c3]

def MW11(m,gene):
    f1 = variablelinkage(m,gene) * plainf1(gene)
    f2 = variablelinkage(m,gene) * numpy.sqrt(2-numpy.power(f1/variablelinkage(m,gene),2))
    c1 = (3-numpy.power(f1,2)-f2)*(3-2*numpy.power(f1,2)-f2)
    c2 = -(3-0.625*numpy.power(f1,2)-f2)*(3-7*numpy.power(f1,2)-f2)
    c3 = (1.62-0.18*numpy.power(f1,2)-f2)*(1.125-0.125*numpy.power(f1,2)-f2)
    c4 = -(2.07-0.23*numpy.power(f1,2)-f2)*(0.63-0.07*numpy.power(f1,2)-f2)
    return [f1,f2],[c1,c2,c3,c4]

def MW12(m,gene):
    f1 = biased(m,gene) * plainf1(gene)
    f2 = biased(m,gene) * (0.85-0.8*(f1/biased(m,gene))-0.08*numpy.abs(numpy.sin(3.2*numpy.pi*(f1/biased(m,gene)))))
    T1 = 1-0.8*f1-f2+0.08*numpy.sin(2*numpy.pi*(f2-(f1/1.5)))
    T2 = 1-0.625*f1-f2+0.08*numpy.sin(2*numpy.pi*(f2-(f1/1.6)))
    T3 = 1.4-0.875*f1-f2+0.08*numpy.sin(2*numpy.pi*(f2/1.4 - (f1/1.6)))
    T4 = 1.8-1.125*f1-f2+0.08*numpy.sin(2*numpy.pi*(f2/1.8 - (f1/1.6)))
    c1 = -(T1*T4)
    c2 = T2 * T3
    return [f1,f2],[c1,c2]

def MW13(m,gene):
    f1 = multimodal(m,gene) * numpy.power(plainf1(gene),len(gene))
    f2 = multimodal(m,gene) * (5-(f1/multimodal(m,gene))-0.5*numpy.abs(numpy.sin(3*numpy.pi*(f1/multimodal(m,gene)))))
    T1 = 5- numpy.exp(f1)-0.5*numpy.sin(3*numpy.pi*f1) - f2
    T2 = 5 - (1+f1+0.5*numpy.power(f1,2)) - 0.5 * numpy.sin(3 * numpy.pi * f1)-f2
    T3 = 5 - (1+0.7*f1) - 0.5 * numpy.sin(3 * numpy.pi * f1)-f2
    T4 = 5 - (1+0.4*f1) - 0.5 * numpy.sin(3 * numpy.pi * f1)-f2
    c1 = - T1 * T4
    c2 = T2 * T3
    return [f1,f2],[c1,c2]

def MW14(m,gene):
    f = []
    for num in gene[:m-1]:
        f.append(num)
    fmbase = [6-numpy.exp(fk)-1.5*numpy.sin(1.1*numpy.pi*numpy.power(fk,2)) for fk in f]
    f.append((variablelinkage(m,gene)/(m-1)) * numpy.sum(numpy.array(fmbase)))
    alpha = [6.1-(1+fk+0.5*numpy.power(fk,2)+1.5*numpy.sin(1.1*numpy.pi*numpy.power(fk,2))) for fk in f]
    c = numpy.sum(numpy.array(alpha))/(m-1) - f[-1]
    return f,[c]

if __name__ == "__main__":
    gene = [0 for i in range(0,10)]
    m = 2
    k = 3
    print(MW1(m,gene))
    print(MW2(m,gene))
    print(MW3(m,gene))
    print(MW4(k,gene))
    print(MW5(m,gene))
    print(MW6(m,gene))
    print(MW7(m,gene))
    print(MW8(k,gene))
    print(MW9(m,gene))
    print(MW10(m,gene))
    print(MW11(m,gene))
    print(MW12(m,gene))
    print(MW13(m,gene))
    print(MW14(k,gene))